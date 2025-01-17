from ibfuncs import get_financials, get_ib
with get_ib("SNP") as ib:
    d = ib.run(get_financials(ib))
    print(d)

