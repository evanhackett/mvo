# Uncorrelated's Mean Variance Optimizer

From the thread on Bogleheads: [A mean variance framework for portfolio optimization](https://www.bogleheads.org/forum/viewtopic.php?t=322366)

Here I've added a `requirements.txt` file to make it easier to get running.

## Build Instructions

Create a virtual env if you haven't already:

```bash
python3 -m venv venv
```

Activate the virtual env:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Now you can run the script:

```bash
python mvo_original.py
```

See the linked thread above for more details.

You can add more assets and adjust the expected returns, volatility, correlations, etc.

## CRRA

I also created a script for calculating your CRRA (constant relative risk aversion, aka "gamma").

For help, run:

```bash
python crra.py -h
```


## Credits

Special thanks to Bogleheads user Uncorrelated for producing the original code and [dedicating it to the public domain](https://www.bogleheads.org/forum/viewtopic.php?p=5576429&sid=d55cb5a4ffacf0d43f9d1dd936958b16#p5576429).
