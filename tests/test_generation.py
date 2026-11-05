import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from generate import StackedRandomGenerator


class GenerationTests(unittest.TestCase):
    def test_pfgm_prior_is_independent_of_batch_partition_and_cpu_rng(self):
        for resolution in (32, 64):
            with self.subTest(resolution=resolution):
                shape = [3, 3, resolution, resolution]
                torch.manual_seed(123)
                before = torch.get_rng_state().clone()
                full = StackedRandomGenerator('cpu', [7, 11, 13]).rand_beta_prime(
                    shape, device='cpu', D=128, sigma_max=80)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                split = torch.cat([
                    StackedRandomGenerator('cpu', [seed]).rand_beta_prime(
                        [1, *shape[1:]], device='cpu', D=128, sigma_max=80)
                    for seed in (7, 11, 13)
                ])
                self.assertTrue(torch.equal(full, split))
                scaled = StackedRandomGenerator('cpu', [7, 11, 13]).rand_beta_prime(
                    shape, device='cpu', D=128, sigma_max=40)
                torch.testing.assert_close(scaled, full / 2, rtol=0, atol=0)

    def test_fid_script_separates_logs_and_preserves_existing_images(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/many_fid.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoints = root / 'checkpoints'
            checkpoints.mkdir()
            (checkpoints / 'network-snapshot-005000.pkl').touch()
            (checkpoints / 'other.pkl').touch()
            mock = root / 'torchrun'
            mock.write_text('''#!/bin/bash
if [[ "$*" == *generate.py* ]]; then
  echo "Using sampler: edm"
  printf '%s\\n' "$*" >> "$CALLS"
  for arg in "$@"; do
    case "$arg" in --outdir=*) mkdir -p "${arg#--outdir=}";; esac
  done
else
  echo "Calculating statistics..."
  echo "2.125"
fi
''')
            mock.chmod(0o755)
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}',
                       CALLS=str(root / 'calls'), SEED_START='50000')
            args = ['bash', str(script), str(checkpoints), 'ref.npz',
                    str(root / 'fid.csv'), '50000', '0', '5000',
                    str(root / 'images'), 'edm', '1', '128']
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / 'fid.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['fid'], '2.125')
            self.assertIn('--seeds=50000-99999', (root / 'calls').read_text())
            result = subprocess.run(args, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Output already exists', result.stderr)
            self.assertTrue((root / 'images/kimg_005000').is_dir())


if __name__ == '__main__':
    unittest.main()
