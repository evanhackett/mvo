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

# Updates

## mvo.py

An updated version of the original script with the following additions:

- **Asset selection via fzf** — on startup, a fuzzy-find menu lets you pick which assets to include without editing the code. Requires `fzf` (`brew install fzf`).
- **Allocation table** — a second window shows a table of the optimal asset allocation for each gamma value, so you don't have to visually cross-reference the charts.
- **Live factor assumptions** — factor return assumptions can be loaded from a JSON file instead of being hardcoded (see below).

### Keeping factor assumptions up to date

`fetch_factors.py` downloads Fama-French factor data from [Ken French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) and computes monthly return and volatility assumptions. Results are cached locally for 30 days.

Fetch and save factor assumptions:

```bash
python fetch_factors.py
```

By default this writes `factors.json`. You can override the output path and restrict the sample period:

```bash
python fetch_factors.py --output my_factors.json --start 199001 --end 202412
```

Force a re-download even if the cache is fresh:

```bash
python fetch_factors.py --refresh
```

Then run `mvo.py` with the generated file:

```bash
python mvo.py --factors factors.json
```

Without `--factors`, `mvo.py` falls back to the original hardcoded assumptions.

> **Note:** ITT and LTT (intermediate/long-term treasury) factor assumptions are not available from the French data library. They are included in `factors.json` with hardcoded baseline values — edit the file manually to update them.
