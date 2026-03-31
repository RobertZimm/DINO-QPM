import numpy as np
import pandas as pd


class ConvergenceTracker:
    def __init__(self, window_size: int = 5, threshold: float = 1e-3, val_name: str = "prototype_mean_change"):
        """
        Initializes the ConvergenceTracker using standard deviation over a window.

        Args:
            window_size (int): Number of recent epochs to consider for standard deviation.
            threshold (float): The standard deviation threshold below which convergence is declared.
        """
        self.val_name = val_name
        self.window_size = window_size
        self.threshold = threshold
        self.values = []
        self.converged = False

    def update(self, current_value: float, info: dict = None):
        """
        Updates the tracker with the current value and checks for convergence.

        Args:
            current_value (float): The current value of the monitored quantity.
        """
        print(f"Updating convergence tracker with value: {current_value}")
        self.values.append(current_value)

        if len(self.values) >= self.window_size:
            # Calculate standard deviation of the last 'window_size' values
            recent_values = self.values[-self.window_size:]
            std_dev = np.std(recent_values)
            
            print(f"Standard deviation over the last {self.window_size} values: {std_dev}")

            if std_dev < self.threshold:
                self.converged = True
                print(f"Convergence reached with std deviation {std_dev} below threshold {self.threshold}")
            
            else:
                print(f"Not yet converged: std deviation {std_dev} above threshold {self.threshold}")
        
        else:
            print(f"Not enough values to determine convergence (have {len(self.values)}, need {self.window_size})")
            
        self.handle_update_logging(current_value, info)
        
    def handle_update_logging(self, current_value: float, info: dict):
        """
        Handles logging or printing of update information.

        Args:
            current_value (float): The current value of the monitored quantity.
            info (dict, optional): Additional information to log.
            
        """
        if not hasattr(self, "log_df"):
            self.val_col = f"{self.val_name} (threshold: {self.threshold})"
            self.log_df = pd.DataFrame(columns=[self.val_col] + (list(info.keys())))
            
        info[self.val_col] = current_value
        
        self.log_df = pd.concat([self.log_df, pd.DataFrame([info])], ignore_index=True)
    
    def save_info(self, filepath: str):
        """
        Saves the logged information to a CSV file.

        Args:
            filepath (str): Path to the file where the log should be saved.
        """
        if hasattr(self, "log_df"):
            self.log_df.to_csv(filepath, index=False)
            print(f"Convergence log saved to {filepath}")
        else:
            print("No log data to save.")

    def has_converged(self) -> bool:
        """
        Checks if convergence has been reached.

        Returns:
            bool: True if converged, False otherwise.
        """
        return self.converged

    def get_values(self) -> list[float]:
        """
        Returns all tracked values.
        """
        return self.values