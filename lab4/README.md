Find best hiperparameters:


```
python build_model.py --max_epochs 5 --executions 1

```

Train model with best hiperparameters:
```
python train.py --params models/best_params.json --epochs 80 --batch_size 8

```

Hyperband:
```
python predict.py --model "models/final_model.h5" --alcohol 14.13 --malic_acid 4.1 --ash 2.74 --alcalinity 24.5 --magnesium 96 --total_phenols 2.05 --flavanoids 0.76 --nonflavanoid_phenols 0.56 --proanthocyanins 1.35 --color_intensity 9.2 --hue 0.61 --od280_od315 1.6 --proline 560
```



