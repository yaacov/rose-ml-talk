# ROSE ML Talk

[slides](https://yaacov.github.io/rose-ml-talk/slides/)

## Serve slides:

```bash
python -m http.server -d ./slides/
```

## Demo - ROSE Game

```bash
cd demo1

# run game engine 
podman run --rm --network host -it quay.io/rose/rose-server

# run driver logic
podman run -it --rm --network host \
  -v $(pwd)/good_driver.py:/driver.py:z \
  quay.io/rose/rose-client \
  --driver /driver.py \
  --port 8081
```

## Demo - Perceptron

```bash
cd demo2

python perceptron.py 
```

## Demo - ROSE ML Driver

https://github.com/yaacov/rose-ml-driver
