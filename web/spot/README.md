# HYDRA Spot early access

A standalone, browser-signed UBTC/USDC buy screen. Open `/spot/` in MetaMask or
Rabby's built-in mobile browser. Ordinary mobile Safari does not inject a wallet;
the screen provides a MetaMask deep link instead. WalletConnect is not configured.

Only Hyperliquid **spot** USDC is spendable here. UBTC is Unit Bitcoin, not a native
Bitcoin UTXO. Funding and withdrawal remain on Hyperliquid/Unit; this page does not
bridge, transfer, approve API agents, request token allowances or store keys.

The first version caps each order at 12–1,000 USDC and uses a single buy IOC order,
with at most 0.5% slippage from the fresh best ask. Exact decimal arithmetic rounds
size and price down. Unfilled amounts cancel; partial fills are possible. Fee
reserve uses `userSpotCrossRate`; actual spot fees may be deducted in the base
asset. No builder fee is requested. A quote expires after 30 seconds, including
wallet confirmation time. Each purchase requires its own wallet signature.

Hyperliquid L1 signatures use the **Exchange / Agent** EIP-712 domain and chain ID
**1337**, even for mainnet. Some injected wallets reject a domain differing from
their active network. We do not add a fictitious network or silently authorize an
agent to work around this. Such wallets can connect and preview, and receive a
clear signing compatibility error. Offline signature tests do not establish
compatibility with every real mobile wallet; this is part of the phone trial.

An uncertain submission is persisted by public client order ID in localStorage,
never by key or signature. New orders are blocked until status reconciliation.
No exchange request is retried automatically. Clearing browser storage removes
that local protection; users must check Hyperliquid history before resubmitting.

## Build and verify

Requires Node 24. From this directory:

```
npm ci --ignore-scripts
npm test
npm run build
```

The locked build writes committed files to `static/spot/`, so the existing Python
Docker image requires no Node runtime. FastAPI serves that directory at `/spot/`.
The GitHub Pages workflow builds a separate HTTPS phone preview from the feature
branch. It requires repository Pages access and deployment permission. This does
not update the Railway service.

Tests use generated, unfunded, ephemeral identities and mocked wallets. Automated
tests must never submit an exchange request or fund a wallet. Public `/info`
requests are used separately to verify live metadata, fees and order books.
