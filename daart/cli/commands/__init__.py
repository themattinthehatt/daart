"""Command modules for the beast CLI."""

from daart.cli.commands import predict, train

# dictionary of all available commands
COMMANDS = {
    'train': train,      # model training
    'predict': predict,  # model inference on keypoints
}
