import pytest
import torch
import numpy as np
from omicstl.deep_learning_utils import *

class TestComputeMSE:
    def test_compute_mse_perfect(self):
        predictions = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        assert compute_mse(predictions, target) == 0

    def test_compute_mse_known(self):
        predictions = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 4.0, 6.0])

        expected_mse = (1**2 + 2**2 + 3**2) / 3
        assert abs(compute_mse(predictions, target) - expected_mse) < 0.0001

    def test_compute_mse_single(self):
        predictions = torch.tensor([1.0])
        target = torch.tensor([2.0])

        expected_mse = 1
        assert abs(compute_mse(predictions, target) - expected_mse) < 0.0001

    def test_compute_mse_negative_perfect(self):
        predictions = torch.tensor([-1.0, -2.0, -3.0])
        target = torch.tensor([-1.0, -2.0, -3.0])

        assert compute_mse(predictions, target) == 0

    def test_compute_mse_negative_known(self):
        predictions = torch.tensor([-1.0, -2.0, -3.0])
        target = torch.tensor([-2.0, -4.0, -6.0])

        expected_mse = (1**2 + 2**2 + 3**2) / 3
        assert abs(compute_mse(predictions, target) - expected_mse) < 0.0001

    def test_compute_mse_mixed_sign_known(self):
        predictions = torch.tensor([-1.0, 2.0, -3.0])
        target = torch.tensor([-2.0, 4.0, -6.0])

        expected_mse = (1**2 + 2**2 + 3**2) / 3
        assert abs(compute_mse(predictions, target) - expected_mse) < 0.0001

    # Don't try to test on systems which don't have CUDA
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_compute_mse_cuda(self):
        # Both on GPU
        predictions = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        target = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        assert compute_mse(predictions, target) == 0

        # Only predictions on GPU
        predictions = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        target = torch.tensor([1.0, 2.0, 3.0])
        assert compute_mse(predictions, target) == 0

        # Only target on GPU
        predictions = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        assert compute_mse(predictions, target) == 0

    def test_compute_mse_with_gradients(self):
        # Both gradients
        predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        predictions = predictions * 2.0
        target = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = target * 2.0
        assert compute_mse(predictions, target) == 0

        # Only predictions have gradients
        predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        predictions = predictions * 2.0
        target = torch.tensor([2.0, 4.0, 6.0])
        assert compute_mse(predictions, target) == 0

        # Only target has gradients
        predictions = torch.tensor([2.0, 4.0, 6.0])
        target = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = target * 2.0
        assert compute_mse(predictions, target) == 0

    def test_compute_mse_with_differing_dtypes(self):
        predictions = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        assert abs(compute_mse(predictions, target)) < 0.0001

    def test_compute_mse_two_dimensional(self):
        predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert compute_mse(predictions, target) == 0

    def test_compute_mse_scalar(self):
        predictions = torch.tensor(1.0)
        target = torch.tensor(2.0)

        expected_mse = 1
        assert abs(compute_mse(predictions, target) - expected_mse) < 0.0001

    def test_compute_mse_empty(self):
        predictions = torch.tensor([])
        target = torch.tensor([])
        with pytest.warns(RuntimeWarning):
            assert np.isnan(compute_mse(predictions, target))

class TestComputeAccuracy:
    def test_compute_accuracy_perfect(self):
        predictions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target = torch.tensor([0, 1, 2])
        
        expected_acc = 1.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    def test_compute_accuracy_wrong(self):
        predictions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target = torch.tensor([2, 2, 0])
        
        expected_acc = 0.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001
        
    def test_compute_accuracy_known(self):
        predictions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        target = torch.tensor([1, 1, 1])
        
        expected_acc = 1/3
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    # Given a toss-up (two classes with exactly 0.5), the lowest class should be chosen
    def test_compute_accuracy_tossup(self):
        predictions = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
        target = torch.tensor([0, 1, 2])
        
        expected_acc = 1/3
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    def test_marginal_wins(self):
        predictions = torch.tensor([[0.33334, 0.33333, 0.33333], [0.33333, 0.33334, 0.33333], [0.33333, 0.33333, 0.33334]])
        target = torch.tensor([0, 1, 2])
        
        expected_acc = 1.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    def test_binary_accuracy_known(self):
        predictions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        target = torch.tensor([0, 0, 1])
        
        expected_acc = 2/3
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    def test_one_prediction(self):
        predictions = torch.tensor([[0.0, 1.0, 0.0]])
        target = torch.tensor([1])
        
        expected_acc = 1.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001
    
    def test_one_class(self):
        predictions = torch.tensor([[1.0], [1.0]])
        target = torch.tensor([0])
        
        expected_acc = 1.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    def test_many_predictions(self):
        pred_template = [0.6, 0.2, 0.2]
        predictions = [pred_template]
        target = [0]
        for i in range(100):
            predictions.append(pred_template)
            target.append(0)
        
        predictions = torch.tensor(predictions)
        target = torch.tensor(target)

        expected_acc = 1.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    def test_compute_accuracy_with_differing_dtypes(self):
        predictions = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]], dtype=torch.float32)
        target = torch.tensor([2, 1], dtype=torch.int64)

        expected_acc = 1.0
        assert abs(compute_accuracy(predictions, target) - expected_acc) < 0.0001

    # Don't try to test on systems which don't have CUDA
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_compute_accuracy_cuda(self):
        # Both on GPU
        predictions = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]], device="cuda")
        target = torch.tensor([2, 1], device="cuda")
        assert compute_mse(predictions, target) == 0

        # Only predictions on GPU
        predictions = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]], device="cuda")
        target = torch.tensor([2, 1])
        assert compute_mse(predictions, target) == 0

        # Only target on GPU
        predictions = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.5, 0.3]])
        target = torch.tensor([2, 1], device="cuda")
        assert compute_mse(predictions, target) == 0

    def test_commute_accuracy_inf(self):
        try:
            predictions = torch.tensor([[0.1, 0.3, np.inf], [0.2, np.inf, 0.3]])
            target = torch.tensor([2, 1])

            compute_accuracy(predictions, target)
        except ValueError:
            return
        
        raise ValueError("Failed to catch infinity")

    def test_commute_accuracy_nan(self):
        try:
            predictions = torch.tensor([[0.1, 0.3, np.nan], [0.2, np.nan, 0.3]])
            target = torch.tensor([2, 1])

            compute_accuracy(predictions, target)
        except ValueError:
            return
        
        raise ValueError("Failed to catch NaN")
