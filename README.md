# Program for optimized sowing, covering, protecting, reaping and de-orphaning SNPs

## TO-DO
- [ ] AAPL, QQQ and SPY chains - only one is being generated. Find out why?
- [ ] `analysis.py` fails when there is no df_chains.pkl.
- [ ] Clean-up `analysis.py` for consistent html and console messages.
- [x] In `analysis.py` check if sowed reward adjusts based on reapratio

## Identifying states

There are three dataframes viz: pf, df_openords and df_unds.
Each of them have the following fields:

- symbol: for name of the symbol
- secType: with STK for stock, OPT for option
- right: with P for put and C for call. Only secType == 'OPT' will have right.
- action: with SELL or BUY
- position: an integer that can be positive or negative

### Portfolio state

Portfolio states are derived from dataframe 'pf'

**Note**: Portfolio has 'state' field.

    - 'zen': Perfect. Stock with both covering and protecting option positions

    - 'exposed': Stock positions without any covering or protecting options
    - 'unprotected': Stock with only covering option position
    - 'uncovered': Stock with only protecting options position

    - 'straddled': Matching call and put options with no underlying stock
      - ... straddles are for stocks having earnings declaration within naked time horizon

    - 'covering': Short calls or puts with underlying stock
    - 'protecting': Long calls or puts with underlying stock
    - 'sowed': Short options without matching stock positions
    - 'orphaned': Long options without matching stock position

### Order state

Order states are derifed from df_openords

    - 'covering' - an option order symbol with action: SELL, having an underlying stock position derived from pf dataframe
    - 'protecting' - an option order symbol with action: BUY, having an underlying stock position derived from pf dataframe
    - 'sowing' - an option order with action: SELL, having no underlying stock position
    - 'reaping' - an option order with action: BUY, having an underlying option position for the same right and strike
    - 'straddling' - two option orders with action: BUY for the same symbol, not in portfolio position
    - 'de-orphaning' - an option order with action: SELL having no underlying stock or any option position

### Symbol States

* Symbol state are derived from portfolio state and order state. They are reflected in df_unds.

  - 'zen': symbol
    - has a stock both covering and protecting portfolio positions or orders
    - has 'straddled' portfolio state
    - has a short 'sowing' order
    - is in 'unprotected' portfolio state with a 'protecting' order
    - is in 'uncovered' portfolio state with a 'covering' order
    - has long option 'orphaned' position with an open 'de-orphaning' order
    - has short option 'sowed' position with a on open 'reaping' order
  - 'unreaped': Symbol has a short option position with no open 'reaping' order
  - 'exposed': Symbol has a stock, but has not covering or protecting order or option position
  - 'uncovered': Symbol has a stock that is protected, but not covered
  - 'unprotected': Symbol has a stock that is covered, but not protected
  - 'virgin': Symbol is not sowed and ready for naked orders
  - 'orphaned': Symbol has a put or call buy position, but without any underlying
  - 'unknown': Anything that is not in any one of the above states. (Should not be there!!)

# Steps

## Get the base ready

* Build `unds` with price and volatility (vy)
* Update `unds` states based on portfolio and open orders
* Get `chains`
* Get `portfolio` and `openorders`. Classify them

## Generate orders

...with correct xPrice

* Make weekly `cover` orders for `exposed` and `uncovered` stock positions
* Make monthly-horizon naked `sow` orders for `virgin` symbols
* Make `reap` orders for unreaped sows
* Make monthly `portect` orders for `unprotected` stock positions
* Make `de-orphan` orders for `orphaned` options

## Check and Place orders

### Analyze

* Find out the account state from get_financials()
* Find out symbols missing in unds from chains, missing in chains from unds, missing in chains from pf
* Find out the costs for protection and how much risk it covers
* Find out maximum and mininmum gains from covers
* Find out premiums from sowing orders
* Find out possible gains from reaping orders
* Find out gains from parenting orders

### Place orders

* Place all orders
