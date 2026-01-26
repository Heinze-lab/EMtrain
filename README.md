# EMtrain

Pipeline for training convolutional neural networks to segment neurons from volume electron microscopy data.
Based on [local-shape descriptors](https://github.com/funkelab/lsd).

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Start a new training run:

```bash
python emtrain/train.py -p <PROJECT_DIR> -cfg <CONFIG_JSON> [--gpu-id 0] [-c NUM_WORKERS]
```

Resume an existing experiment:

```bash
python emtrain/train.py -p <PROJECT_DIR> -cfg <CONFIG_JSON> -r <EXPERIMENT_ID>
```

Options:
- `-p, --project-dir`: Path to the project directory
- `-cfg, --config`: Path to training configuration JSON
- `--gpu-id`: GPU device ID (default: 0)
- `-c, --cache-workers`: Number of cache workers for data loading
- `-r, --resume`: Experiment ID to resume
- `--no-comet-log`: Disable Comet ML logging

### Model Evaluation

Evaluate trained models against ground truth annotations:

```bash
python evaluate_models.py
```

## Configuration

Training and evaluation are controlled via JSON configuration files. Examples are provided in:

- `training_config.json` - Training parameters
- `ground_truth_config.json` - Ground truth dataset paths
- `seg_config.json` - Segmentation pipeline configuration
- `volumes config JSONs` - Per-volume evaluation configurations

## Author

Valentin Gillet (valentin.gillet@biol.lu.se)

## License

MIT License.

## Acknowledgments

EMtrain is built on [gunpowder](https://github.com/funkelab/gunpowder/tree/main) and [local-shape descriptors](https://github.com/funkelab/lsd) by the Funke lab. 
Also see the publication associated with local-shape descriptors: [Local shape descriptors for neuron segmentation](https://www.nature.com/articles/s41592-022-01711-z)