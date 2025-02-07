# lmkit

## Setup

```
# get uv

curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
# clone repository
git clone https://github.com/leakedweights/lmkit.git
cd lmkit

# setup environment
uv venv
source .venv/bin/activate
```

```
uv sync --extra cuda # if working with a gpu
uv sync # otherwise
```