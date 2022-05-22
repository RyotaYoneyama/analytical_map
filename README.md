# analytical_map

Calculate mAP in respect to categories, boounding box size.


## Requirements

## Preparation
- Clone
```
git clone https://github.com/RyotaYoneyama/analytical_map.git
cd analytical_map
```

- docker
```
xhost+
chmod +x docker/entrypoint.sh
docker build -t analytical_map -f docker/Dockerfile .
docker run --rm -it --privileged --net=host --ipc=host \
    -v $PWD:/home/$(id -un)/analytical_map/ \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
    -e XAUTHORITY=/home/$(id -un)/.Xauthority \
    -e DOCKER_USER_NAME=$(id -un) \
    -e DOCKER_USER_ID=$(id -u) \
    -e DOCKER_USER_GROUP_NAME=$(id -gn) \
    -e DOCKER_USER_GROUP_ID=$(id -g) \
    analytical_map bash
```

## Usage
### Evaluate every images and crate a middle file.
Count all TPs, FPs, FNs and categorize them followings:
- TPs->'Match', 
- FPs->'DC (Double Count)', 'Cls (Class mistake)', 'Loc(Location)', 'Bkg (Background)'
- FNs-> 'LC (Less Count)', 'Miss'

```
python3 -m analytical_map.cocoEvaluator
```
The above command will create the middle file named 'sample_results/middle_file.json' and visualization in 'sample_results/visualize'.

### Analyze the middle file and calculate AP, precision, and recall.
```
python3 -m analytical_map.cocoAnalizer
```
The above command will create the final results named 'sample_results/final_results.json' containing scores of AP, precision, and recall.
The charts of those scores are dipicted in 'sample_results/'.

### Evaluate & Analyze
If you would like to execute the above two steps, you can run them by follwoings:
```
python3 -m analytical_map.main COCO_GT_FILE_PATH COCO_DT_FILE_PATH RESULT_DIR IMAGE_DIR
```
For example,
```
python3 -m analytical_map.main sample_data/coco/gt.json sample_data/coco/dt.json sample_results sample_data/images/
```

## Use flow chart
![Use flow chart](docs/figures/use_flow.drawio.png)

## Details
docstring -> (docs/_build/index.html)


## Project status
In progress
