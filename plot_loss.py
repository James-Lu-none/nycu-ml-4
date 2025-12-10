import os
import matplotlib.pyplot as plt


def plot_and_save_loss(log_history, out_dir, title="Loss"):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    for log in log_history:
        if "loss" in log and "step" in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])

        if "eval_loss" in log and "step" in log:
            val_steps.append(log["step"])
            val_losses.append(log["eval_loss"])
    
    plt.figure()
    plt.plot(train_steps, train_losses)
    plt.title(f"{title} - Train")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.close()

    if len(val_losses) > 0:
        plt.figure()
        plt.plot(val_steps, val_losses)
        plt.title(f"{title} - Validation")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "val_loss.png"))
        plt.close()

    print("Loss plots saved to", out_dir)
