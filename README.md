# Executing

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py --input_data <train/file/path> --epochs <num_epochs>
```

Example:

```bash
python train.py --input_data ./datasets/input.txt --epochs 10
```

## Continue training

```bash
python continue_training.py --input_data <train/file/path> --epochs <num_epochs> --saved_model </path/to/model> --output_dir <default/models>
```

> Note: After Continued training the model is saved in the `output_dir`, which is set to `models/` by default.

## Test the Model

```bash
python test.py --model_path </path/to/model>
```

Examples:

Testing a trained model

```bash
python test.py --model_path model.pth
```

Testing a trained model after continued training

```bash
python test.py --model_path .\models\model_1.pth
```
