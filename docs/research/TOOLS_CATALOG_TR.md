# Wallet tracker, likidasyon haritası ve veri kaynakları

29 Ekim 2026. **16 ürün sayfası ve 3 bağımsız açık kaynak araç** incelendi.
Bu tarama Pain dışındaki seçenekleri ve HYDRA'ya ne katabileceklerini kapsar.

En somut veri araştırması adayları **0xArchive, Bitquery ve Hydromancer**.
Ekran/ürün karşılaştırmasında **HyperTracker, HyperDash, ASXN ve CryptoAnimon**
öne çıkıyor. Bunlar ölçülmüş alfa veya kârlılık sıralaması değildir.

Kanıt düzeyi aşağıdaki tablolarda ayrıdır: **sağlayıcı kaynağı**, yayıncının
dokümanı/kodu; **bağımsız kaynak**, üçüncü taraf client/araç; **dizin**, topluluk
listesindeki ürün tanımıdır. Dizin kaydı ürünün veri kalitesini doğrulamaz.
11 README commit ve SHA-256 ile sabitlendi; iki notebook ve üç GraphQL sorgusu
ayrıca incelenip aynı commit'teki dosyalarla eşleştirildi. Servis API denemesi,
plan/fiyat, kapsam ve gecikme benchmarkı yapılmadı. Kaynaklar aşağıdadır.

## Wallet ve trader sayfaları

| Araç | Kaynakta tanımlanan kullanım | HYDRA için katkı | Kanıt ve sınır |
|---|---|---|---|
| [HyperTracker](https://hypertracker.io/) | Canlı wallet/whale takibi, filtreli trader listeleri | Daha geniş adres keşfi; seçim tarihini kaydederek yeni cohort kurma | Dizin [S2]. Gerçek kapsama sayısı, API ve gecikme ölçülmedi |
| [HyperDash](https://hyperdash.com/) | Wallet cohort'ları, whale uyarıları, Copy Score ve copy-trading terminali | Açma/ekleme/azaltma/kapatmayı ayrı inceleme | Dizin [S2]. Copy Score'un ileriye dönük başarısı ölçülmedi; aynı isimli GitHub projeleri ticari ürün sayılmadı |
| [HypurrScan](https://hypurrscan.io/) | Hyperliquid explorer ve hesap/işlem dashboard'u | Ham işlem ve adres kontrolü | Dizin [S2]. Üçüncü taraf monitor botu resmî arayüz yerine geçirilmedi |
| [ASXN Hyperscreener](https://hyperscreener.asxn.xyz/) | Perps, spot, HIP-3, gelir ve top-trader analytics | Cüzdan yönünü daha geniş piyasa bağlamına koyma | Dizin [S2]. Metriklerin geçmişi ve kaynakları ayrıca doğrulanmalı |
| [CryptoAnimon](https://cryptoanimon.com/en/) | Wallet round/PnL, live liquidation map, cüzdan işlem uyarıları | Pain benzeri harita ile wallet görünümünü birlikte karşılaştırma | Dizin [S1]. Haritanın gerçekleşmiş fill mi tahmini risk seviyesi mi gösterdiği doğrulanmalı |
| [HyperAlerts](https://hyperliquid.tv/) | Fiyat, funding, OI, wallet, büyük işlem ve HIP-3 uyarıları | Olay kuyruğu ve değişiklik özetinin ürün tasarımı | Dizin [S1]. Güncel ücretsiz kapsam/plan denenmedi |
| [TraderSpy](https://traderspy.app/) | Hyperliquid trader pozisyon/PnL, diğer borsa bağlamı, orderbook ve paper trading | Cüzdan hareketi ile borsa akışını karşılaştırma | Dizin [S1]. Borsalar arası kimlik eşlemesi ve veri kapsamı ölçülmedi |
| [VaultVision](https://vaultvision.tech/vaults/scanner) | Vault taraması, drawdown/risk, TVL, para akışı ve uyarılar | Tek dönem ROI yanında risk ve sermaye akışını izleme | Dizinler [S1–S2]. Vault getirisi, cüzdan kopyalama getirisi değildir |
| [PerpLedger](https://swarm-t3.github.io/perpledger/) | Adresin yıllık realized PnL, ücret/funding ve CSV görünümü | Transfer, kapalı PnL, açık PnL, ücret ve funding ayrımı | Dizin [S1]. Muhasebe ve sayfalama kodu bağımsız doğrulanmadı |
| [Nansen](https://www.nansen.ai/) | Cohort, wallet profiler, etiketler, perp pozisyon/PnL komutları | Etiketli trader araştırması ve cohort analizi | Önceden okunan sağlayıcı referansları [N1]. Bugünkü kazanan etiketi geçmişe taşınamaz |

Bu araçlar adres keşfi ve olay incelemesi için adaydır. Bugünkü PnL sıralamasını
geçmişte biliniyormuş gibi kullanmak sağ kalan/seçim yanlılığı üretir. Kaynakta
“smart money” yazması, dış hedge'lerin veya ortak hesap sahiplerinin bilindiği
anlamına gelmez. Mevcut iki donmuş örneklem bu tarama nedeniyle değiştirilmedi.

## Veri ve türev piyasası kaynakları

| Kaynak | Gerçekte incelenen | HYDRA'nın hangi eksiğine aday? | Erişim/kanıt sınırı |
|---|---|---|---|
| [0xArchive](https://www.0xarchive.io/) | Sağlayıcının liquidation notebooku; `liquidations.history`, cursor pagination, `trade_id`, zaman/fiyat/miktar/direction ve candle history [S3] | 22 hesaptaki örneklemin dışına çıkarak daha fazla gerçek liquidation olayında H1'i test etme | API anahtarı gerekiyor. Örnekler Free/son 30 gün olarak tanımlanıyor; hesap kotası ve eksiksizlik denenmedi |
| [Bitquery Hyperliquid](https://docs.bitquery.io/docs/perpetuals/hyperliquid/) | `PerpLiquidations`, L3 `BookUpdates`, TWAP, funding ve pozisyon sorguları; üç örnek GraphQL dosyası [S5] | Gerçekleşmiş liquidation, parent/child TWAP ve emir yerleştirme/iptal/dolum analizi | Token gerekiyor. README GraphQL için yaklaşık 30 gün, daha uzun tarih için data lake/Parquet yolu tanımlıyor; bütün piyasa kapsamı sağlayıcı beyanı, ölçülmüş sonuç değil |
| [Hydromancer](https://hydromancer.xyz/) | SDK'da L4 orderbook, `best_bid_users`, `orders_by_user`, kullanıcı likiditesi ve USD depth metotları [S4] | Zorunlu akış sonrası maker tepkisi ve defter absorpsiyonu | API anahtarı gerekiyor. SDK'daki latency değerleri sağlayıcı örneği; bizim benchmarkımız değil. Reservoir arşivinin fiyat/kapsam koşulları doğrulanmadı |
| [Coinalyze](https://coinalyze.net/) | Bağımsız client'ta future markets, predicted funding ve OI history örnekleri [S8] | Wallet hipotezine karşı funding/OI piyasa baseline'ı | Client resmî sağlayıcı deposu değil. Symbol/exchange kapsamı, birimler, quota ve ücret denenmeli |
| [Velo Data](https://velo.xyz/) | Bağımsız client; cross-exchange funding, premium, OI, volume delta ve liquidation serileri [S9] | Spot/perp ve borsalar arası akış karşılaştırması | İncelenen client belgelenmemiş chart endpointlerini kullanıyor. Destekli resmî API ayrı; bu client doğrudan üretime taşınmadı |
| [Vextorium](https://vextorium.com/) | Python client; funding, Hyperliquid whale/liquidation-map ve regime servisleri [S10] | Native public API üzerindeki ek veri değerini karşılaştırma | “API key yok” ücretsiz demek değil: client x402 ile Base USDC ödemesi yapıyor. Ödeme/servis çağrısı yapılmadı |

0xArchive liquidation notebookunda kayıtlar cursor ile tamamlanıyor ve trade ID
ile tekilleştiriliyor. Bu yararlı bir ingest örneği. Aynı sağlayıcının funding
notebooku ise birim ölçeğini oran büyüklüğünden tahmin eden bir kontrol ve tüm
örneklem ortalama/std'siyle çizilen eşikler içeriyor. Birim metadata'dan
doğrulanmalı; bu görselleştirmeler kilitlenmiş OOS strateji kanıtı sayılmamalı.

## Açık kaynakta alınabilecek tasarım fikirleri

| Proje | İncelenen fikir | Kullanım sınırı |
|---|---|---|
| [hl-liqmap](https://github.com/akagifreeez/hl-liqmap) [S6] | Public trade tape'ten adres keşfi, `liquidationPx` envanteri, OI kapsamı, görünür L2 derinlik ve cascade stress | Pozisyon snapshot'ındaki risk seviyesi gerçekleşmiş liquidation değildir. Sığ defter yenilenir; stress sonucu fiyat tahmini değildir |
| [Hyperliquid Trader Tracker](https://github.com/KonScanner/hyperliquid-trader-tracker) [S7] | Public `trades.users` filtresi, open/add/reduce/close state machine, başlangıç pozisyonunu sessiz seed etme, tek olay kartını güncelleme | Cold-start tasarımı yararlı. README pozisyonları bellekte tuttuğunu söylüyor; güvenilir geçmiş için ayrı as-of journal gerekir. Repo çalıştırılmadı, bildirim gönderilmedi |
| [hl-liquidation-heatmap](https://github.com/yodablocks/hl-liquidation-heatmap) [S11] | OI düşüşünden `markPx × OI_delta` ile heatmap üretme | **Gerçek liquidation kanıtı olarak kullanılmamalı.** OI gönüllü kapanışta da düşer. README'nin bütün public API'lerde liquidation flag olmadığı iddiası bizim kaydettiğimiz explicit liquidation nesneleriyle uyuşmuyor |

Yerel public `userFillsByTime` kaydında `liquidation` nesnesinin `liquidatedUser`,
`markPx`, `method` alanları kontrol edildi. Örnek kaydın kendi direction'ı
`Open Long` idi; bu, izlenen hesabın karşı taraf olabileceğini gösterir.
“Close Long” filtresi veya OI düşüşü tek başına victim/gerçek liquidation tanımı
olamaz. [Kişi/adres içermeyen kontrol özeti](evidence/ecosystem-20261029/native-liquidation-proof.json).

## Alfa araştırması için somut öncelik

1. **Gerçekleşmiş liquidation + spot absorpsiyonu:** 0xArchive veya Bitquery'den
   alınacak deneme verisini native kayıtlarımızla aynı `(coin, trade_id)` veya
   sağlayıcının eşdeğer benzersiz event anahtarıyla eşleştirmek. Aynı fill'in
   victim/counterparty ve farklı sağlayıcı raporlarını tekrar toplamayız.
2. **TWAP akışı:** Bitquery'nin parent TWAP ve child-fill kapsamını araştırmak.
   Bilinen kalan program ile gerçekten yürütülen akış ayrı tutulur; iptal edilen
   veya henüz dolmamış emir geleceğin kesin alım/satımı değildir. Bu yeni bir
   araştırma hipotezidir; geçmişten başarı çıkarılmış bir strateji değildir.
3. **Maker/likidite cevabı:** Hydromancer L4 adres/likidite görünümü veya
   `hl-liqmap`'in defter basıncı fikriyle zorunlu akışa karşı gerçek absorpsiyonu
   ölçmek. Defter iptalleri ile fill'ler ayrı olaylardır.
4. **Trader davranış geçmişi:** HyperTracker/HyperDash/ASXN gibi keşif ekranlarını
   cohort adayları için incelemek; etiketi ve keşif tarihini dondurmak. Mevcut
   22 hesaba sonuçlara bakarak yeni kazanan eklenmez.

Her veri denemesinde event zamanı, sağlayıcı observed/late bilgisi, bizim receipt
zamanımız, yönün kime ait olduğu, baz/USD birimi, funding aralığı, sayfalama,
boşluk ve tekrarlar kontrol edilmelidir. Görsel olarak iyi bir harita bunların
yerine geçmez. Maliyet, gecikme ve bağımsız ileri dönem olmadan alfa kabul edilmez.

## Kaynak ve erişim kaydı

Kaynak düzeyinde araştırma tamamlandı. Bazı doğrudan ürün URL'leri proxy 403
verdi; bu nedenle yeni bir panel kullanımı veya güncel fiyat doğrulaması iddiası
yok. Erişim kaydı ve makine tarafından okunabilen 19 giriş:
[katalog](evidence/ecosystem-20261029/catalog.json),
[doğrudan erişim](evidence/ecosystem-20261029/direct-access.json),
[sabit README commit/hash'leri](evidence/ecosystem-20261029/pinned-sources.json),
[notebook/sorgu commit/hash'leri](evidence/ecosystem-20261029/pinned-details.json).

- **S1:** [sbs2001/awesome-hyperliquid](https://github.com/sbs2001/awesome-hyperliquid). Topluluk dizini.
- **S2:** [dexlytrade/awesome-hyperliquid-copy-trading](https://github.com/dexlytrade/awesome-hyperliquid-copy-trading). Dexly ekibinin tuttuğu dizin; bağımsız ürün benchmarkı değildir.
- **S3:** [0xArchiveIO/examples](https://github.com/0xArchiveIO/examples).
- **S4:** [hydromancerxyz/hydromancer-sdk](https://github.com/hydromancerxyz/hydromancer-sdk).
- **S5:** [bitquery/hyperliquid-api](https://github.com/bitquery/hyperliquid-api).
- **S6:** [akagifreeez/hl-liqmap](https://github.com/akagifreeez/hl-liqmap).
- **S7:** [KonScanner/hyperliquid-trader-tracker](https://github.com/KonScanner/hyperliquid-trader-tracker).
- **S8:** [ivarurdalen/coinalyze](https://github.com/ivarurdalen/coinalyze). Bağımsız client.
- **S9:** [0xAtasoy/veloxyz-scraper](https://github.com/0xAtasoy/veloxyz-scraper). Bağımsız client.
- **S10:** [bru1noia/vextorium-python](https://github.com/bru1noia/vextorium-python).
- **S11:** [yodablocks/hl-liquidation-heatmap](https://github.com/yodablocks/hl-liquidation-heatmap). Bağımsız araç.
- **N1:** [Nansen komut referansı](https://github.com/nansen-ai/nansen-skills/blob/main/references/commands.md); önceki incelemede okundu.

Bu aşama belge/kod kaynağı incelemesidir. Uygulama kodu, donmuş cüzdanlar, gerçek
emir ve çalışan veri toplayıcılar değiştirilmedi. Kaynak snapshot'ları
`/workspace/hydra-ecosystem-20261029` altında yerel olarak korunuyor.
