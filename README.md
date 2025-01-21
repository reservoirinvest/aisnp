# Program for optimized sowing, covering, protecting, reaping and de-orphaning SNPs

## Identifying states

### Portfolio state

**Note**: Portfolio has 'state' field.

    - 'zen': Perfect. Stock with both covering and protecting option positions

    - 'exposed': Stock positions without options
    - 'unprotected': Stock with only covering option position
    - 'uncovered': Stock with only protecting options position
    - 'straddled': Matching call and put options with no underlying stock
      - ... straddles are for stocks having earnings declaration within naked time horizon
    - 'covering': Short calls or puts with underlying stock
    - 'protecting': Long calls or puts with underlying stock
    - 'sowed': Short options without matching stock positions
    - 'orphaned': Long options without matching stock position

### Order state

    - 'covering' - an option order symbol with action: SELL, having an underlying stock position derived from pf dataframe
    - 'protecting' - an option order symbol with action: BUY, having an underlying stock position derived from pf dataframe
    - 'sowing' - an option order with action: SELL, having no underlying stock position
    - 'reaping' - an option order with action: BUY, having an underlying option position for the same right and strike
    - 'straddling' - two option orders with action: BUY for the same symbol, not in portfolio position
    - 'de-orphaning' - an option order with action: SELL having no underlying stock or any option position

### Symbol States

* Symbol state are derived from portfolio state and order state. They are reflected in df_unds.

  - 'zen': symbol
    - has both covering and protecting portfolio positions or orders
    - has 'straddled' portfolio state
    - has a short 'sowing' order
    - is in 'unprotected' portfolio state with a 'protecting' order
    - is in 'uncovered' portfolio state with a 'covering' order
    - has long option 'orphaned' position with an open 'de-orphaning' order
    - has short option 'sowed' position with a on open 'reaping' order
  - 'unreaped': Symbol has a short option position with no open 'reaping' order
  - 'virgin': Symbol is not sowed and ready for naked orders

  ...rest would be state derived from position

# Steps

## Get the base ready

* Get `portfolio` and `openorders`. Classify them
* Build `unds` with price and volatility (vy)
* Update `unds` states based on portfolio and open orders 

* Get `chains`
  - **Note:** Chain generation is to be done with **IBG** (not TWS)

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
