import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from tqdm import tqdm
from scipy.sparse import csr_matrix


class CliqueDataset(Dataset):
    def __init__(self, base_dataset, k=10, min_clique_size=2, max_clique_size=10):
        self.base_dataset = base_dataset
        self.k = k
        self.min_clique_size = min_clique_size
        self.max_clique_size = max_clique_size
        
        print(f"Building kNN graph with k={k}...")
        tmp_loader = DataLoader(base_dataset, batch_size=128, num_workers=0, shuffle=False)
        all_data = []
        all_images_cache = []
        all_labels_cache = []
        
        for batch in tqdm(tmp_loader, desc="Loading data"):
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            elif isinstance(batch, np.ndarray):
                images = torch.from_numpy(batch).float()
                labels = None
            else:
                images = batch
                labels = None
            
            all_data.append(images.flatten(1).cpu())
            all_images_cache.append(images.cpu())
            if labels is not None:
                all_labels_cache.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.from_numpy(labels))
        
        all_data = torch.cat(all_data, dim=0).numpy()
        self.cached_images = torch.cat(all_images_cache, dim=0)
        self.cached_labels = torch.cat(all_labels_cache, dim=0) if all_labels_cache else None
        
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1).fit(all_data)
        distances, indices = nbrs.kneighbors(all_data)
        
        edges = []
        for i in tqdm(range(len(indices)), desc="Building edges"):
            for j in indices[i][1:]:
                if i < j:
                    edges.append((i, j))
        
        G = nx.Graph()
        G.add_edges_from(edges)
        
        print("Finding maximal cliques...")
        all_cliques = list(nx.find_cliques(G))

        self.all_cliques = [
            c for c in all_cliques 
            if self.min_clique_size <= len(c) <= self.max_clique_size
        ]
        print(f"Filtered to {len(self.all_cliques)} cliques (size {self.min_clique_size}-{self.max_clique_size})")
        
        self.node_to_cliques = [[] for _ in range(len(base_dataset))]
        for clique_idx, clique in enumerate(tqdm(self.all_cliques, desc="Mapping nodes to cliques")):
            for node in clique:
                self.node_to_cliques[node].append(clique_idx)
        
        self.valid_nodes = [i for i, cliques in enumerate(self.node_to_cliques) if len(cliques) > 0]
        
        print("Calculating optimal clique weights for uniform node distribution...")
        num_nodes = len(base_dataset)
        node_degrees = np.array([len(self.node_to_cliques[v]) for v in range(num_nodes)], dtype=float)
        node_degrees[node_degrees == 0] = 1.0
        
        weights = np.array([
            np.sum(1.0 / node_degrees[list(c)]) 
            for c in tqdm(self.all_cliques, desc="Computing initial weights")
        ])
        
        target_node_prob = 1.0 / len(self.valid_nodes)
        for _ in range(1):
            node_probs = np.zeros(num_nodes)
            for c_idx, clique in enumerate(self.all_cliques):
                for v_idx in clique:
                    node_probs[v_idx] += weights[c_idx]
            
            node_corr = np.ones(num_nodes)
            mask = node_probs > 0
            node_corr[mask] = target_node_prob / node_probs[mask]
            
            for c_idx, clique in enumerate(self.all_cliques):
                corr = np.mean([node_corr[v] for v in clique])
                weights[c_idx] *= corr
        
        self.clique_weights_approx = weights / weights.sum()
        self.clique_weights = self.compute_uniform_weights(self.all_cliques, len(self.base_dataset))
        
        print(f"Using {len(self.all_cliques)} cliques. {len(self.valid_nodes)} nodes covered.")

    def compute_uniform_weights(self, cliques, num_nodes, max_iter=50, tol=1e-6):
        row_ind = []
        col_ind = []
        for c_idx, c in enumerate(tqdm(cliques, desc="Building sparse matrix")):
            for v in c:
                row_ind.append(v)
                col_ind.append(c_idx)
        data = np.ones(len(row_ind))
        A = csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, len(cliques)))

        w = self.clique_weights_approx.copy()

        pbar = tqdm(range(max_iter), desc="Computing uniform weights")
        for _ in pbar:
            node_probs = A.dot(w) + 1e-8
            log_factors = np.log(1.0 / node_probs)
            clique_log_scale = np.zeros(len(cliques))
            for c_idx, c in enumerate(cliques):
                clique_log_scale[c_idx] = np.mean([log_factors[v] for v in c])
            w *= np.exp(clique_log_scale)
            w /= w.sum()
            cv = np.std(node_probs) / np.mean(node_probs)
            pbar.set_postfix({"CV": f"{cv:.6f}"})
            if cv < tol:
                break
        return w

    def __len__(self):
        return len(self.all_cliques)

    def __getitem__(self, idx):
        clique = self.all_cliques[idx]
        
        clique_images = [self.cached_images[i] for i in clique]
        clique_labels = [self.cached_labels[i] if self.cached_labels is not None else None for i in clique]
            
        return {
            "images": clique_images,
            "labels": clique_labels,
            "all_nodes": list(clique)
        }


class WeightedCliqueSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        indices = torch.multinomial(self.weights, self.num_samples, replacement=self.replacement)
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def clique_collate_fn(batch):
    all_images = []
    all_labels = []
    all_nodes = []
    
    for sample in batch:
        all_images.extend(sample["images"])
        all_labels.extend(sample["labels"])
        all_nodes.extend(sample["all_nodes"])
    
    images = torch.stack(all_images, dim=0)
    
    if all_labels[0] is not None:
        labels = torch.stack([
            torch.from_numpy(l) if isinstance(l, np.ndarray) else l 
            for l in all_labels
        ])
    else:
        labels = None
    
    return np.array(all_nodes), images, labels


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class CliqueDataLoader:
    def __init__(self, base_dataset, k, min_clique_size, max_clique_size, batch_size, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.cl_dataset = CliqueDataset(
            base_dataset, k=k, 
            min_clique_size=min_clique_size, 
            max_clique_size=max_clique_size
        )
        
        avg_clique_size = np.mean([len(c) for c in self.cl_dataset.all_cliques])
        self.cliques_per_batch = max(1, int(np.ceil(batch_size / avg_clique_size * 1.5)))
        
        num_batches = len(base_dataset) // batch_size
        self.num_samples_per_epoch = num_batches * self.cliques_per_batch
        
        self.sampler = WeightedCliqueSampler(
            weights=self.cl_dataset.clique_weights,
            num_samples=self.num_samples_per_epoch,
            replacement=True
        )
        
        self._loader = DataLoader(
            self.cl_dataset,
            batch_size=self.cliques_per_batch,
            sampler=self.sampler,
            collate_fn=clique_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    def __iter__(self):
        for node_indices, images, labels in self._loader:
            n = len(node_indices)
            if n < self.batch_size:
                continue
            
            perm = np.random.permutation(n)[:self.batch_size]
            
            batch_images = images[perm]
            batch_indices = node_indices[perm]
            
            if labels is not None:
                batch_labels = labels[perm]
            else:
                batch_labels = torch.zeros(self.batch_size, dtype=torch.long)
            
            yield batch_indices, batch_images, batch_labels

    def __len__(self):
        return len(self._loader)
