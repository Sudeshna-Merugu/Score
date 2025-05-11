import logging
from typing import Any, Dict, List
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    """
    PyTorch Dataset class for the MBPP (Mostly Basic Python Programming) dataset.
    Each sample contains a task prompt, solution code, and test cases.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        """
        Args:
            data: A list of dictionaries, each representing a single MBPP sample.
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.data[idx]
            # Return standardized keys
            return {
                "text": item.get("text", idx),
                "code": item["code"],
                "test_list": item.get("test_list", [])
            }

        except KeyError as e:
            logger.error(f"Missing expected key in sample at index {idx}: {e}")
            raise
        except IndexError as e:
            logger.error(f"Index {idx} out of range for dataset of size {len(self.data)}.")
            raise
        except Exception as e:
            logger.error(f"Error retrieving item at index {idx}: {e}")
            raise
