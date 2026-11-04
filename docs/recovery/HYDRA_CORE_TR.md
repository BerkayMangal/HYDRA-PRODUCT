# HYDRA ana ürününü toparlama — 4 Kasım 2026

Kullanıcı yönlendirmesi: önce asıl HYDRA tek ürün olarak tamamlanacak. Spot alım,
cüzdan bağlama ve builder gelir özelliği sonraya bırakıldı. Ayrı Pages denemesi,
asıl Railway sitesine entegrasyonun tamamlandığı anlamına gelmiyor.

## Korunacak mevcut üretim

Canlı adres: https://web-production-57fc0.up.railway.app/

- HYDRA + VENOM ana arayüzü; mevcut Radar, Markets, Funding ve Proof akışları.
- `/api/scalp` içindeki üç BTC/ETH modeli. Çalışma durumu 4 Kasım kontrolünde aktif.
- Mevcut veri ve model defterleri. Eski checkout ile değiştirilmemeli.

Çalışan `/assets/hydra.js`, `/api/scalp` kaynakları ve üç `config/models/*.joblib`
modeli bu repo sürümünde yok. Tüm dallarda ve geçmişte yüklenen ZIP dosyasının
ad listesinde de model kaynakları bulunmadı. Public GitHub profilinde ikinci bir
HYDRA reposu bulunamadı. Bu gözlemler canlı servisin kaynağını belirlemez;
Railway kaynak/deployment bilgisi hâlâ gerekli.

## Geliştirmeler kaybolmadı

| Bileşen | Kaynak / kayıt | Asıl sitede 4 Kasım durumu |
| --- | --- | --- |
| Veri/cache/ETF/OI/funding ve kalite düzeltmeleri | `74bb83c`, `tests/test_data_integrity.py`, `main.py` | Çalışan kaynakla aynı sürüm oldukları doğrulanmadı |
| Public hesap ve akış gözlemleri | `services/wallet_monitor.py`, `74bb83c`, `955f15d` | `/api/wallets`: 404 |
| Araştırma/backtest kanıt paneli | `research/dashboard_summary.py`, `static/index.html` | `/api/research`: 404 |
| Ücret/kayma maliyet senaryoları | `research/cost_scenarios.py`, `955f15d` | `/api/research/costs`: 404 |
| Kaynak/rakip kataloğu | 19 kaynaklık katalog, `/api/research/tools` | 404; bu katalog ticari servis entegrasyonu değil |
| İleri TWAP gözlemleri ve kayıt defteri | `2a0fe7a` → `c4a1672`, `research/forward_ledger.py` | `/api/research/forward`: 404 |
| Maliyet ve yoğunlaşma ayrıştırması | `research/forward_diagnostics.py`, `e88b904` | Repo içinde; canlı entegrasyonu tamamlanmadı |
| Likidasyon veri pilotu | `4708fdf` → `1292280`, `research/liquidation_pilot.py` | `/api/research/pilot`: 404 |
| Sürüm/dağıtım doğrulaması | `scripts/railway_deploy.py`, `f5a43e5` | `/api/version`: 404 |
| Spot cüzdan bağlantısı / alım | `web/spot`, ayrı Pages denemesi | Ana ürün kapsamından ertelendi; `/spot/`: 404 |

Kaynaklar `feature/spot-wallet-preview-20261103`,
`research/liquidation-pilot-20261102`, `research/twap-forward-20261031` dallarında
ve Git geçmişinde korunuyor. Bir kısmı GitHub main'de de mevcut; canlı uygulama
ile eşleşmiyor. Geliştirmeler kârlı bir strateji bulunduğu iddiası taşımıyor.

## Eski panelin gerçek kayıtları

Bunlar **30 Ekim tarihli geliştirme doğrulama görüntüleri**, yeni üretim yayını
değil. Güncel fiyat veya sonuç gibi kullanılmamalı:

- [Araştırma ve kaynak paneli](evidence/20261104/research-panel-20261030.png)
- [Maliyet senaryosu paneli](evidence/20261104/cost-scenarios-20261030.png)
- [4 Kasım canlı route kontrolü](evidence/20261104/production-routes.json)

## Tek ürün için taşıma sırası

1. Railway yönetim panelinden mevcut servisin repo/dal/kök dizini, aktif deployment
   SHA'sı veya manuel upload kaynağını belirle. GitHub'ın main dalını değiştirmek
   tek başına üretime bağlandığını kanıtlamaz.
2. Çalışan güncel kaynak ve model dosyalarını ayrı bir çalışma dalında koru.
   Kaynak GitHub dışından yüklenmişse önce gerçek kaynak dosyaları gerekli.
3. Yukarıdaki hazır analiz ve kalite modüllerini mevcut modele ve navigasyona
   birleştir. Eski dashboard ile bütün uygulamayı değiştirme. Aynı bilgiyi üreten
   mevcut Radar/Proof modüllerini kopyalayarak iki ayrı sonuç defteri yaratma.
4. Birleşik sürümü test et; eksik ham veriyi tamamlanmış veya güncel veri olarak
   göstermeden eski araştırmaları tarihli kanıt olarak sun.
5. Önce geri dönüşü olan staging doğrulaması, sonra mevcut Railway servisine
   yayın ve aynı asıl URL'de sürüm, ekran ve veri kontrolleri. Bu son kontrol
   yapılmadan ana HYDRA entegrasyonu tamamlandı sayılmaz.

Bugün eklenen Railway source audit yalnızca metadata okur; uygulama değişkeni,
log, komut veya hesap anahtarı okumaz. Upload, restart, sipariş ve bildirim yok.
`[deployment-audit]` commit'leri normal Python CI'dan geçer; Railway dağıtımını
başlatmaz. [Kaynak erişimi kontrolü](https://github.com/BerkayMangal/HYDRA-PRODUCT/actions/runs/41807286338)
GitHub repository + production environment kapsamından çalıştı ve
`missing_railway_access` döndürdü. Akışın Success olması yalnız tanı işleminin
tamamlandığını gösterir; gerçek dağıtım sayısı sıfırdır. Sonuç
[evidence/20261104/railway-source-access.json](evidence/20261104/railway-source-access.json)
içinde. Bir sonraki gerekli bilgi, https://railway.com/dashboard üzerinden doğru
projenin yönetim ekranıdır; uygulamanın public URL'si bu bilgiyi sağlamaz.
