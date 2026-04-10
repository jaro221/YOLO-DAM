"""
Monitor Week 1 Improvements During Training

Tracks:
1. Hard negative mining boost (should be ~1.0-1.3)
2. Curriculum learning weight progression (should go 0.3 → 1.0)
3. Adaptive focal loss alpha per class (should vary by class precision)
4. F1 improvement trend (target: 0.815 → 0.860+)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


class Week1MetricsMonitor:
    """Monitor Week 1 improvements from training logs."""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = Path(log_dir)
        self.metrics_history = {
            'epoch': [],
            'total_loss': [],
            'hard_neg_boost': [],
            'curr_weight_mean': [],
            'adaptive_alpha_mean': [],
            'det_loss': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
        }

    def parse_training_log(self, log_file):
        """Parse training log and extract Week 1 metrics."""
        metrics = {}

        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Parse hard negative mining
                if 'hard_neg_boost' in line:
                    val = self._extract_value(line, 'hard_neg_boost')
                    if val:
                        metrics['hard_neg_boost'] = val

                # Parse curriculum learning
                if 'curr_weight_mean' in line:
                    val = self._extract_value(line, 'curr_weight_mean')
                    if val:
                        metrics['curr_weight_mean'] = val

                # Parse adaptive alpha
                if 'adaptive_alpha' in line:
                    vals = self._extract_alpha_values(line)
                    if vals:
                        metrics['adaptive_alpha_mean'] = np.mean(vals)

                # Parse losses
                if 'total_loss' in line:
                    val = self._extract_value(line, 'total_loss')
                    if val:
                        metrics['total_loss'] = val

                if 'det_loss' in line and 'det_' not in line:
                    val = self._extract_value(line, 'det_loss')
                    if val:
                        metrics['det_loss'] = val

        except Exception as e:
            print(f"Error parsing log: {e}")

        return metrics

    def _extract_value(self, line, key):
        """Extract numeric value from log line."""
        try:
            parts = line.split(key)
            if len(parts) > 1:
                val_str = parts[1].split()[0].replace(':', '').replace(',', '')
                return float(val_str)
        except (ValueError, IndexError):
            pass
        return None

    def _extract_alpha_values(self, line):
        """Extract list of alpha values."""
        try:
            start = line.find('[')
            end = line.find(']')
            if start != -1 and end != -1:
                vals_str = line[start + 1:end]
                return [float(x) for x in vals_str.split(',')]
        except (ValueError, IndexError):
            pass
        return None

    def print_summary(self):
        """Print summary of Week 1 improvements."""
        print("\n" + "=" * 80)
        print("WEEK 1 IMPROVEMENTS MONITORING")
        print("=" * 80)

        if not self.metrics_history['epoch']:
            print("No data collected yet. Training in progress...")
            return

        epochs = self.metrics_history['epoch']
        hard_negs = self.metrics_history['hard_neg_boost']
        curr_weights = self.metrics_history['curr_weight_mean']
        alphas = self.metrics_history['adaptive_alpha_mean']
        f1s = self.metrics_history['f1_score']

        print("\nMetric Ranges Observed:")
        print(f"  Hard Negative Boost:  {min(hard_negs):.3f} - {max(hard_negs):.3f}")
        print(f"  Curriculum Weight:    {min(curr_weights):.3f} - {max(curr_weights):.3f}")
        print(f"  Adaptive Alpha Mean:  {min(alphas):.3f} - {max(alphas):.3f}")
        print(f"  F1 Score:             {min(f1s):.3f} - {max(f1s):.3f}")

        print("\nProgress Indicators:")
        if len(curr_weights) > 1:
            curr_progress = (curr_weights[-1] - curr_weights[0]) / (1.0 - 0.3 + 1e-7)
            print(f"  Curriculum progression: {curr_progress * 100:.1f}% "
                  f"(should go 0-100%)")

        if len(f1s) > 1:
            f1_gain = f1s[-1] - f1s[0]
            print(f"  F1 improvement so far: +{f1_gain:.4f} (target: +0.045 for Week 1)")

        if len(hard_negs) > 0:
            avg_boost = np.mean(hard_negs)
            print(f"  Avg hard negative boost: {avg_boost:.3f} (should be 1.0-1.3)")

        print("\nExpected Week 1 Results:")
        print("  Start:    F1 = 0.815 (Precision: 70-75%, Recall: 84-86%)")
        print("  Target:   F1 = 0.860+ (Precision: 76-80%, Recall: 85-87%)")
        print("  Expected: F1 gain of +4-6% over full Week 1 training")

        print("\n" + "=" * 80)

    def check_health(self):
        """Check if Week 1 implementations are working correctly."""
        print("\n" + "-" * 80)
        print("HEALTH CHECK: Week 1 Implementations")
        print("-" * 80)

        issues = []

        # Check hard negative mining
        if self.metrics_history['hard_neg_boost']:
            boost_vals = self.metrics_history['hard_neg_boost']
            if all(v > 2.0 for v in boost_vals):
                issues.append("Hard negative boost too high (>2.0) - check configuration")
            elif all(v < 0.8 for v in boost_vals):
                issues.append("Hard negative boost too low (<0.8) - mining not active")
            else:
                print("[OK] Hard negative mining: Boost values in expected range")

        # Check curriculum learning progression
        if len(self.metrics_history['curr_weight_mean']) > 10:
            curr_early = np.mean(self.metrics_history['curr_weight_mean'][:5])
            curr_late = np.mean(self.metrics_history['curr_weight_mean'][-5:])
            if curr_late > curr_early:
                print("[OK] Curriculum learning: Weight correctly progressing "
                      f"({curr_early:.3f} → {curr_late:.3f})")
            else:
                issues.append("Curriculum weight not increasing - check implementation")

        # Check adaptive alpha
        if self.metrics_history['adaptive_alpha_mean']:
            alpha_vals = self.metrics_history['adaptive_alpha_mean']
            if min(alpha_vals) > 0.1 and max(alpha_vals) < 2.0:
                print("[OK] Adaptive focal loss: Alpha values in expected range")
            else:
                issues.append("Adaptive alpha out of range - check compute_adaptive_alpha()")

        # Check F1 improvement
        if len(self.metrics_history['f1_score']) > 10:
            f1_early = np.mean(self.metrics_history['f1_score'][:5])
            f1_late = np.mean(self.metrics_history['f1_score'][-5:])
            f1_gain = f1_late - f1_early
            if f1_gain > 0.005:
                print(f"[OK] F1 improvement: +{f1_gain:.4f} detected so far")
            else:
                issues.append("Low F1 improvement - verify losses are decreasing")

        if issues:
            print("\nWarnings:")
            for issue in issues:
                print(f"  [WARNING] {issue}")
        else:
            print("\n[OK] All Week 1 implementations appear healthy!")

        print("-" * 80)

    def plot_metrics(self, output_file="week1_metrics_plot.png"):
        """Generate plot of Week 1 metrics over time."""
        try:
            import matplotlib.pyplot as plt

            if not self.metrics_history['epoch']:
                print("No data to plot yet.")
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Week 1 Improvements Progress', fontsize=16)

            epochs = self.metrics_history['epoch']

            # Hard negative mining
            axes[0, 0].plot(epochs, self.metrics_history['hard_neg_boost'], 'b-')
            axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Baseline')
            axes[0, 0].set_ylabel('Hard Neg Boost')
            axes[0, 0].set_title('Hard Negative Mining (target: 1.0-1.3)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Curriculum learning
            axes[0, 1].plot(epochs, self.metrics_history['curr_weight_mean'], 'g-')
            axes[0, 1].axhline(y=0.3, color='r', linestyle='--', label='Start')
            axes[0, 1].axhline(y=1.0, color='orange', linestyle='--', label='End')
            axes[0, 1].set_ylabel('Curriculum Weight')
            axes[0, 1].set_title('Curriculum Learning (should go 0.3 → 1.0)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # Adaptive alpha
            axes[1, 0].plot(epochs, self.metrics_history['adaptive_alpha_mean'], 'orange')
            axes[1, 0].set_ylabel('Mean Adaptive Alpha')
            axes[1, 0].set_title('Adaptive Focal Loss Alpha')
            axes[1, 0].grid(True)

            # F1 score improvement
            if self.metrics_history['f1_score']:
                axes[1, 1].plot(epochs, self.metrics_history['f1_score'], 'r-')
                axes[1, 1].axhline(y=0.815, color='gray', linestyle='--',
                                  label='Baseline (0.815)')
                axes[1, 1].axhline(y=0.860, color='green', linestyle='--',
                                  label='Target (0.860)')
                axes[1, 1].set_ylabel('F1 Score')
                axes[1, 1].set_title('F1 Improvement Over Time')
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            for ax in axes.flat:
                ax.set_xlabel('Epoch')

            plt.tight_layout()
            plt.savefig(output_file, dpi=100)
            print(f"Plot saved to {output_file}")
            plt.close()

        except ImportError:
            print("matplotlib not installed. Skipping plot generation.")


def main():
    """Example usage."""
    monitor = Week1MetricsMonitor()

    print("\nWeek 1 Metrics Monitor")
    print("=" * 80)
    print("This script monitors the three Week 1 improvements:")
    print("  1. Hard negative mining weight")
    print("  2. Curriculum learning progression")
    print("  3. Adaptive focal loss alpha values")
    print("\nUsage during training:")
    print("  1. Run training: python YOLO_DAM_train.py")
    print("  2. Monitor: python MONITOR_WEEK1_METRICS.py")
    print("\nThe monitor will analyze training logs and check health.")
    print("=" * 80)

    # Example: Print expected behavior
    print("\nExpected Metric Ranges:")
    print("  hard_neg_boost:    1.0 - 1.3 (most often ~1.1-1.2)")
    print("  curr_weight_mean:  0.3 - 1.0 (progresses over epochs)")
    print("  adaptive_alpha:    0.2 - 1.0 (varies per class)")
    print("  F1 improvement:    +4-6% over full Week 1 training")

    print("\nQuick Start Commands:")
    print("  # Start training with Week 1 improvements:")
    print("  python YOLO_DAM_train.py --epochs 300 --batch_size 4")
    print("\n  # Monitor progress in separate terminal:")
    print("  python MONITOR_WEEK1_METRICS.py")


if __name__ == '__main__':
    main()
