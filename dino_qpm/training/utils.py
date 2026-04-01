
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# from robustness.robustness.tools.helpers  https://github.com/MadryLab/robustness


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class VariableLossLogPrinter():
    def __init__(self):
        self.losses = {}

    def log_loss(self, key, val, n=1):
        if not key in self.losses:
            self.losses[key] = AverageMeter()
        self.losses[key].update(val, n)

    def get_loss_string(self):
        loss_string = " | ".join(
            [f"{key}: {self.losses[key].avg:.4f}" for key in self.losses])

        return loss_string


class TrainingLogger:
    """
    A comprehensive logging class that handles CSV logging and visualization of training metrics.
    """

    def __init__(self, log_dir: str, mode: str):
        """
        Initialize the training logger.

        Args:
            log_dir: Directory to save logs and visualizations
            mode: Training mode (e.g., 'dense', 'finetune')
        """
        self.log_dir = log_dir
        self.mode = mode
        self.csv_file = None
        self.csv_writer = None
        self.csv_fieldnames = None
        self.csv_file_handle = None

        if log_dir and log_dir != '':
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # Create graphs subdirectory
            self.viz_dir = os.path.join(log_dir, 'graphs')
            os.makedirs(self.viz_dir, exist_ok=True)

            # Initialize CSV file
            self._init_csv()

    def _init_csv(self):
        """Initialize CSV file for logging."""
        if not self.log_dir:
            return

        self.csv_file = os.path.join(
            self.log_dir, f'training_logs_{self.mode}.csv')

        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(
            self.csv_file) and os.path.getsize(self.csv_file) > 0

        # Open file in append mode
        try:
            self.csv_file_handle = open(self.csv_file, 'a', newline='')

            # If file exists and has content, read existing fieldnames
            if file_exists:
                try:
                    import pandas as pd
                    existing_df = pd.read_csv(
                        self.csv_file, nrows=0)  # Read only headers
                    self.csv_fieldnames = list(existing_df.columns)
                    self.csv_writer = csv.DictWriter(
                        self.csv_file_handle,
                        fieldnames=self.csv_fieldnames
                    )
                except Exception:
                    # If reading fails, we'll initialize fieldnames on first write
                    pass

        except Exception as e:
            print(f"Warning: Failed to open CSV file {self.csv_file}: {e}")
            return

    def log(self, logs: Dict[str, Any], epoch: int, phase: str = "train") -> None:
        """
        Log training metrics to CSV file.

        Args:
            logs: Dictionary containing all training logs
            epoch: Current epoch
            phase: Training phase (train/test)
        """
        if not self.log_dir or not self.csv_file_handle:
            return

        # Add metadata to logs
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'mode': self.mode,
            'phase': phase
        }

        # Add all log values
        for key, value in logs.items():
            log_entry[key] = value

        try:
            # Initialize CSV writer if not done yet
            if self.csv_writer is None:
                self.csv_fieldnames = list(log_entry.keys())
                self.csv_writer = csv.DictWriter(
                    self.csv_file_handle,
                    fieldnames=self.csv_fieldnames
                )

                # Write header if file is new/empty
                if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
                    self.csv_writer.writeheader()
            else:
                # If new fields are added, we need to handle them
                new_fields = set(log_entry.keys()) - set(self.csv_fieldnames)
                if new_fields:
                    # Close current file and reinitialize with new fieldnames
                    self._reinit_csv_with_new_fields(log_entry)

            # Write the log entry
            self.csv_writer.writerow(log_entry)
            self.csv_file_handle.flush()  # Ensure data is written immediately

        except Exception as e:
            print(f"Warning: Failed to write to CSV file {self.csv_file}: {e}")

    def _reinit_csv_with_new_fields(self, log_entry: Dict[str, Any]):
        """Reinitialize CSV with new fieldnames when new fields are detected."""
        if not self.csv_file_handle:
            return

        try:
            # Close current file
            self.csv_file_handle.close()

            # Read existing data
            existing_data = []
            if os.path.exists(self.csv_file):
                try:
                    df = pd.read_csv(self.csv_file)
                    existing_data = df.to_dict('records')
                except Exception:
                    pass  # If reading fails, start fresh

            # Update fieldnames
            self.csv_fieldnames = list(log_entry.keys())

            # Reopen file in write mode to add new headers
            self.csv_file_handle = open(self.csv_file, 'w', newline='')
            self.csv_writer = csv.DictWriter(
                self.csv_file_handle,
                fieldnames=self.csv_fieldnames
            )

            # Write header and existing data
            self.csv_writer.writeheader()
            for row in existing_data:
                # Fill missing fields with None
                for field in self.csv_fieldnames:
                    if field not in row:
                        row[field] = None
                self.csv_writer.writerow(row)

            self.csv_file_handle.flush()

        except Exception as e:
            print(f"Warning: Failed to reinitialize CSV with new fields: {e}")

    def close(self):
        """Close the CSV file handle."""
        if self.csv_file_handle:
            self.csv_file_handle.close()
            self.csv_file_handle = None
            self.csv_writer = None

    def visualize_training(self, save_plots: bool = True) -> None:
        """
        Create visualizations of training metrics and optionally save them.

        Args:
            save_plots: Whether to save plots to the graphs directory
        """
        if not self.log_dir or not os.path.exists(self.csv_file):
            print("No CSV file found for visualization")
            return

        try:
            # Read the CSV data
            df = pd.read_csv(self.csv_file)

            if df.empty:
                print("No data found in CSV file")
                return

            print(f"Debug - CSV file: {self.csv_file}")
            print(f"Debug - Mode: {self.mode}")

            # Set up matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create visualizations
            self._plot_accuracy_and_loss(df, save_plots)
            self._plot_regularization_losses(df, save_plots)
            self._plot_training_overview(df, save_plots)

            if save_plots:
                print(f"Graphs saved to: {self.viz_dir}")

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    def plot_single_metric(self, metric_name: str, save_plot: bool = True, show_plot: bool = True):
        """
        Plot a single specific metric by name.
        This method is not called automatically - use it when you want to visualize a specific metric.

        Args:
            metric_name: Name of the metric to plot
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
        """
        if not self.log_dir or not os.path.exists(self.csv_file):
            print("No CSV file found for visualization")
            return

        try:
            df = pd.read_csv(self.csv_file)

            if df.empty:
                print("No data found in CSV file")
                return

            if metric_name not in df.columns:
                available_metrics = [col for col in df.columns
                                     if col not in {'timestamp', 'epoch', 'mode', 'phase'}]
                print(
                    f"Metric '{metric_name}' not found. Available metrics: {available_metrics}")
                return

            # Set up matplotlib style
            plt.style.use('default')
            sns.set_palette("husl")

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            train_mask = df['phase'] == 'train'
            test_mask = df['phase'] == 'test'

            # Plot train data
            if train_mask.any() and df[metric_name].notna().any():
                train_data = df[train_mask]
                if not train_data.empty:
                    ax.plot(train_data['epoch'], train_data[metric_name],
                            label='Train', marker='o', linewidth=2)

            # Plot test data
            if test_mask.any() and df[metric_name].notna().any():
                test_data = df[test_mask]
                if not test_data.empty:
                    ax.plot(test_data['epoch'], test_data[metric_name],
                            label='Test', marker='s', linewidth=2)

            ax.set_title(
                f'{metric_name} Over Time - {self.mode.title()} Mode', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)

            # Only show legend if we have data to plot
            has_train_data = train_mask.any() and df[metric_name].notna().any()
            has_test_data = test_mask.any() and df[metric_name].notna().any()
            if has_train_data or has_test_data:
                ax.legend()

            plt.tight_layout()

            if save_plot:
                filename = f'{self.mode}_{metric_name.replace("/", "_").replace(" ", "_").replace("-", "_")}.png'
                plt.savefig(os.path.join(self.viz_dir, filename),
                            dpi=300, bbox_inches='tight')
                print(f"Plot saved: {filename}")

            if show_plot:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"Error plotting {metric_name}: {e}")

    def list_available_metrics(self) -> List[str]:
        """
        Get a list of all available metrics that can be plotted.

        Returns:
            List of metric names available for plotting
        """
        if not self.log_dir or not os.path.exists(self.csv_file):
            return []

        try:
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return []

            # Return all columns except metadata
            metadata_cols = {'timestamp', 'epoch', 'mode', 'phase'}
            return [col for col in df.columns if col not in metadata_cols]

        except Exception as e:
            print(f"Error reading metrics: {e}")
            return []

    def _plot_accuracy_and_loss(self, df: pd.DataFrame, save_plots: bool):
        """Plot accuracy and main loss metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f'Training Progress - {self.mode.title()} Mode', fontsize=16)

        # Find accuracy column dynamically
        acc_col = None
        possible_acc_cols = ['Acc', 'accuracy', 'test_acc', 'train_acc', 'acc']
        for col in possible_acc_cols:
            if col in df.columns:
                acc_col = col
                break

        # Plot accuracy
        if acc_col:
            for phase in ['train', 'test']:
                mask = df['phase'] == phase
                if mask.any():
                    axes[0, 0].plot(df[mask]['epoch'], df[mask][acc_col],
                                    label=f'{phase.title()} Accuracy', marker='o')

        axes[0, 0].set_title('Accuracy Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        if acc_col:
            axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Find loss column dynamically
        loss_col = None
        possible_loss_cols = ['CE-Loss', 'loss',
                              'train_loss', 'test_loss', 'cross_entropy_loss']
        for col in possible_loss_cols:
            if col in df.columns:
                loss_col = col
                break

        # Plot cross-entropy loss
        if loss_col:
            for phase in ['train', 'test']:
                mask = df['phase'] == phase
                if mask.any():
                    axes[0, 1].plot(df[mask]['epoch'], df[mask][loss_col],
                                    label=f'{phase.title()} Loss', marker='o')

        axes[0, 1].set_title('Loss Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        if loss_col:
            axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot FDL loss
        fdl_col = 'FDL'
        if fdl_col in df.columns:
            train_mask = df['phase'] == 'train'
            if train_mask.any():
                axes[1, 0].plot(df[train_mask]['epoch'], df[train_mask][fdl_col],
                                label='FDL Loss', marker='o', color='red')

        axes[1, 0].set_title('Feature Diversity Loss Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('FDL Loss')
        if fdl_col in df.columns:
            axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot total loss
        total_col = 'Total-Loss'
        if total_col in df.columns:
            train_mask = df['phase'] == 'train'
            if train_mask.any():
                axes[1, 1].plot(df[train_mask]['epoch'], df[train_mask][total_col],
                                label='Total Loss', marker='o', color='purple')

        axes[1, 1].set_title('Total Loss Over Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Total Loss')
        if total_col in df.columns:
            axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(self.viz_dir, f'{self.mode}_accuracy_loss.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_regularization_losses(self, df: pd.DataFrame, save_plots: bool):
        """Plot regularization losses (L1, FSL, IoU, etc.)."""
        # Find regularization loss columns
        reg_losses = []
        possible_reg_losses = ['L1-FM', 'L1-W',
                               'L1-FV', 'FSL', 'IoU', 'FDL-Avg']

        for loss_type in possible_reg_losses:
            col_name = loss_type
            if col_name in df.columns and df[col_name].notna().any():
                reg_losses.append((loss_type, col_name))

        if not reg_losses:
            return

        # Create subplots
        n_losses = len(reg_losses)
        cols = min(3, n_losses)
        rows = (n_losses + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))

        # Ensure consistent axes handling for different subplot configurations
        if n_losses == 1:
            # Single subplot - make it indexable as 2D
            axes = [[axes]]
        elif rows == 1 and cols > 1:
            # Single row, multiple columns
            axes = [axes]
        elif cols == 1 and rows > 1:
            # Single column, multiple rows
            axes = [[ax] for ax in axes]
        # For rows > 1 and cols > 1, axes is already 2D

        fig.suptitle(
            f'Regularization Losses - {self.mode.title()} Mode', fontsize=16)

        train_mask = df['phase'] == 'train'

        for idx, (loss_name, col_name) in enumerate(reg_losses):
            row = idx // cols
            col = idx % cols

            # Get the correct axis based on the subplot configuration
            if n_losses == 1:
                ax = axes[0][0]  # Single subplot
            elif rows == 1 and cols > 1:
                ax = axes[0][col]  # Single row, multiple columns
            elif cols == 1 and rows > 1:
                ax = axes[row][0]  # Single column, multiple rows
            else:
                ax = axes[row][col]  # Multiple rows and columns

            if train_mask.any() and df[col_name].notna().any():
                ax.plot(df[train_mask]['epoch'], df[train_mask][col_name],
                        label=loss_name, marker='o')
                ax.set_title(f'{loss_name} Loss Over Time')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'{loss_name} Loss')
                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_losses, rows * cols):
            row = idx // cols
            col = idx % cols

            # Get the correct axis for hiding based on the subplot configuration
            if n_losses == 1:
                # No unused plots for single subplot
                continue
            elif rows == 1 and cols > 1:
                axes[0][col].axis('off')  # Single row, multiple columns
            elif cols == 1 and rows > 1:
                axes[row][0].axis('off')  # Single column, multiple rows
            else:
                axes[row][col].axis('off')  # Multiple rows and columns

        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(self.viz_dir, f'{self.mode}_regularization_losses.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_training_overview(self, df: pd.DataFrame, save_plots: bool):
        """Create an overview plot with multiple metrics."""
        print(f"Debug - Available columns: {list(df.columns)}")
        print(f"Debug - DataFrame shape: {df.shape}")
        print(
            f"Debug - Unique phases: {df['phase'].unique() if 'phase' in df.columns else 'No phase column'}")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f'Training Overview - {self.mode.title()} Mode', fontsize=16)

        # Ensure axes is always a 2D array for consistent indexing
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)

        train_mask = df['phase'] == 'train'
        test_mask = df['phase'] == 'test'

        # 1. Train vs Test Accuracy - find accuracy column dynamically
        acc_col = None
        possible_acc_cols = ['Acc', 'accuracy', 'test_acc', 'train_acc', 'acc']
        for col in possible_acc_cols:
            if col in df.columns:
                acc_col = col
                break

        if acc_col and train_mask.any():
            axes[0, 0].plot(df[train_mask]['epoch'], df[train_mask][acc_col],
                            label='Train', marker='o')
        if acc_col and test_mask.any():
            axes[0, 0].plot(df[test_mask]['epoch'], df[test_mask][acc_col],
                            label='Test', marker='s')

        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        if acc_col:
            axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Train vs Test Loss - find loss column dynamically
        loss_col = None
        possible_loss_cols = ['CE-Loss', 'loss',
                              'train_loss', 'test_loss', 'cross_entropy_loss']
        for col in possible_loss_cols:
            if col in df.columns:
                loss_col = col
                break

        if loss_col and train_mask.any():
            axes[0, 1].plot(df[train_mask]['epoch'], df[train_mask][loss_col],
                            label='Train', marker='o')
        if loss_col and test_mask.any():
            axes[0, 1].plot(df[test_mask]['epoch'], df[test_mask][loss_col],
                            label='Test', marker='s')

        axes[0, 1].set_title('CE Loss Comparison (Train vs Test)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('CE Loss')
        if loss_col:
            axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Loss Components Breakdown - find available loss components dynamically
        loss_components = []
        # Look for any columns that might be loss components
        loss_keywords = ['loss', 'Loss', 'FDL', 'CE', 'Total']
        for col in df.columns:
            if any(keyword in col for keyword in loss_keywords) and df[col].notna().any():
                loss_components.append((col, col))

        # Plot training loss components only (test typically only has CE-Loss which isn't useful here)
        for comp_name, col_name in loss_components:
            if train_mask.any():
                axes[0, 2].plot(df[train_mask]['epoch'], df[train_mask][col_name],
                                label=comp_name, marker='o', linestyle='-')

        axes[0, 2].set_title('Loss Components (Training)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        if loss_components:
            axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Epoch timing (if available)
        if 'timestamp' in df.columns:
            try:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
                if train_mask.any():
                    epoch_times = df[train_mask].groupby(
                        'epoch')['timestamp_dt'].first()
                    if len(epoch_times) > 1:
                        time_diffs = epoch_times.diff().dt.total_seconds()
                        axes[1, 0].plot(epoch_times.index[1:], time_diffs[1:],
                                        marker='o', label='Time per Epoch')
                        axes[1, 0].set_title('Training Speed')
                        axes[1, 0].set_xlabel('Epoch')
                        axes[1, 0].set_ylabel('Seconds per Epoch')
                        axes[1, 0].grid(True, alpha=0.3)
            except Exception:
                axes[1, 0].text(0.5, 0.5, 'Timing data unavailable',
                                transform=axes[1, 0].transAxes, ha='center')

        # 5. Learning rate (if available in logs)
        lr_col = 'learning_rate'
        if lr_col in df.columns and train_mask.any():
            axes[1, 1].plot(df[train_mask]['epoch'], df[train_mask][lr_col],
                            marker='o', label='Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning rate data unavailable',
                            transform=axes[1, 1].transAxes, ha='center')

        # 6. Summary statistics
        if train_mask.any() or test_mask.any():
            summary_text = []

            # Best accuracy
            if acc_col:
                test_accs = df[test_mask][acc_col] if test_mask.any(
                ) else df[train_mask][acc_col]
                if len(test_accs) > 0:
                    best_acc = test_accs.max()
                    mask_to_use = test_mask if test_mask.any() else train_mask
                    best_acc_epoch = df[mask_to_use][df[mask_to_use]
                                                     [acc_col] == best_acc]['epoch'].iloc[0]
                    acc_type = 'Test' if test_mask.any() else 'Train'
                    summary_text.append(
                        f'Best {acc_type} Acc: {best_acc:.4f} (Epoch {best_acc_epoch})')

            # Final losses
            if loss_col:
                if train_mask.any():
                    train_losses = df[train_mask][loss_col]
                    if len(train_losses) > 0:
                        final_train_loss = train_losses.iloc[-1]
                        summary_text.append(
                            f'Final Train Loss: {final_train_loss:.4f}')

                if test_mask.any():
                    test_losses = df[test_mask][loss_col]
                    if len(test_losses) > 0:
                        final_test_loss = test_losses.iloc[-1]
                        summary_text.append(
                            f'Final Test Loss: {final_test_loss:.4f}')

            # Total epochs
            max_epoch = df['epoch'].max()
            summary_text.append(f'Total Epochs: {max_epoch}')

            axes[1, 2].text(0.1, 0.7, '\n'.join(summary_text),
                            transform=axes[1, 2].transAxes, fontsize=12,
                            verticalalignment='top')
            axes[1, 2].set_title('Training Summary')
            axes[1, 2].axis('off')

        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(self.viz_dir, f'{self.mode}_training_overview.png'),
                        dpi=300, bbox_inches='tight')
        plt.show()

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.

        Returns:
            Dictionary containing training summary statistics
        """
        if not self.log_dir or not os.path.exists(self.csv_file):
            return {}

        try:
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return {}

            summary = {
                'mode': self.mode,
                'total_epochs': df['epoch'].max(),
                'total_training_steps': len(df[df['phase'] == 'train'])
            }

            # Accuracy statistics
            acc_col = 'Acc'
            if acc_col in df.columns:
                test_mask = df['phase'] == 'test'
                if test_mask.any():
                    test_accs = df[test_mask][acc_col]
                    summary.update({
                        'best_test_accuracy': test_accs.max(),
                        'final_test_accuracy': test_accs.iloc[-1],
                        'best_test_accuracy_epoch': df[test_mask][df[test_mask][acc_col] == test_accs.max()]['epoch'].iloc[0]
                    })

            # Loss statistics
            train_loss_col = 'CE-Loss'
            if train_loss_col in df.columns:
                train_mask = df['phase'] == 'train'
                if train_mask.any():
                    train_losses = df[train_mask][train_loss_col]
                    summary.update({
                        'initial_train_loss': train_losses.iloc[0],
                        'final_train_loss': train_losses.iloc[-1],
                        'min_train_loss': train_losses.min()
                    })

            return summary

        except Exception as e:
            print(f"Error generating training summary: {e}")
            return {}

    @staticmethod
    def load_and_visualize(log_dir: str, mode: str, save_plots: bool = True):
        """
        Load an existing training log and create visualizations.
        Useful for post-training analysis.

        Args:
            log_dir: Directory containing the training logs
            mode: Training mode
            save_plots: Whether to save the generated plots
        """
        logger = TrainingLogger(log_dir, mode)
        logger.visualize_training(save_plots)
        summary = logger.get_training_summary()
        logger.close()
        return summary


def get_prototype_training_logs(
    model,
    alignment_info: Optional[Dict[str, Any]],
    epoch: int,
    mode: str,
    config: dict
) -> Dict[str, Any]:
    """
    Generate logging information for prototype training.

    Args:
        model: Dino2Div model instance
        alignment_info: Information from prototype alignment (if any)
        epoch: Current epoch
        mode: Training mode
        config: Configuration dictionary to get original epochs from

    Returns:
        Dict[str, Any]: Logging dictionary for wandb/tensorboard
    """
    logs = {}

    if not hasattr(model, 'proto_layer') or model.proto_layer is None:
        return logs

    # Basic prototype information
    logs[f"prototype_enabled"] = 1
    logs[f"prototype_count"] = model.proto_layer.n_prototypes
    logs[f"alignment_count"] = getattr(
        model.proto_layer,
        "_alignment_count",
        getattr(model.proto_layer, "_projection_count", 0),
    )

    # Get original epochs from config based on mode
    original_epochs = config.get(mode, {}).get('epochs')

    # Extended training information
    if original_epochs is not None:
        logs[f"original_epochs"] = original_epochs
        logs[f"is_extended_training"] = 1 if epoch >= original_epochs else 0
        if epoch >= original_epochs:
            logs[f"extended_epochs"] = epoch - original_epochs + 1

    # Alignment information if available
    if alignment_info is not None:
        logs[f"proto_mean_distance"] = alignment_info.get(
            'mean_distance', 0)
        logs[f"proto_max_distance"] = alignment_info.get(
            'max_distance', 0)
        logs[f"proto_max_change"] = alignment_info.get(
            'max_change', float('inf'))
        logs[f"proto_mean_change"] = alignment_info.get(
            'mean_change', float('inf'))

    return logs


if __name__ == "__main__":
    # Example usage
    log_dir = Path.home() / "tmp/dinov2" / "Fitzpatrick17k" / \
        "Study1-Split_Method/1470223_9/ft"
    mode = "finetune"
    logger = TrainingLogger(log_dir, mode)

    logger.visualize_training(save_plots=True)
    # logger.plot_single_metric("Acc", save_plot=False, show_plot=True)
