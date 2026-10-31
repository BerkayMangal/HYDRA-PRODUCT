# HYDRA araştırma teslimi ve çalıştırma

30 Ekim güncellemesi: en yeni sürüm/dağıtım komutları için
[dağıtım kılavuzu](../DEPLOYMENT_TR.md). Aşağıdaki eski bitiş örneklerini yeniden
başlatmayın; yeni açık `--until` bütçesi gerekir. Hyperliquid worker artık API
süresini poll aralığına eklemeyen sabit başlangıç zamanlaması kullanır. Eksik
14–18 dakika referansı sıfıra çevrilmez; `inventory_window_status` incelenir.
Güncel kayıt raporları pencere kapsamını da içerir. Native TWAP için
`python -m research.lab.twap_probe` bir defalık salt okunur deneme yapar.

29 Ekim 2026. Bu dosya güncel operasyon kılavuzudur. [Sonuç raporu](FINAL_TR.md)
gerçekte ölçülen bulguları; diğer tarihli raporlar önceki audit ve deneyleri kapsar.

## Kurulum ve doğrulama

Repo kökünde Python 3.12 kullanın:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -p no:cacheprovider
```

Mevcut cloud ortamının Python'u `/workspace/.venvs/hydra/bin/python`.
Public OKX ve Hyperliquid okumaları için özel anahtar gerekmez. Canlı Railway
sürümünün bu checkout ile eşleştiği doğrulanmış değildir.

## Donmuş tarihsel deneyi çevrimdışı geri yükleme

Varsayılan çalışma dizini `data/research/`; `HYDRA_RESEARCH_DIR` ile değişir.
Geri yükleme, sıkıştırılmış ve açılmış her dosyanın SHA-256 değerini kontrol eder.
Farklı bir mevcut dosyayı ezmez. Veri, seçim aşamaları ve cohort birlikte korunur.

```bash
.venv/bin/python -m research.lab.restore
.venv/bin/python -m research.lab.verify
```

`docs/research/evidence/bundle-manifest.json` tam dosya listesidir. Paket:

- 59.802 kapanmış BTC/USDT spot 1H mumu ve veri kalite sözleşmesi.
- Önceden tanımlı 13.368 varyasyon, discovery/validation seçimi ve 4 donmuş aday.
- Saatlik equity serileri, maliyet/gecikme stresleri ve sıfır terfi sonucunun kanıtları.
- PnL ile seçilmemiş 6 public hesabın donmuş cohort ve keşif sözleşmesi.

Tam tarihsel taramanın kaynak kodu `research/lab/search.py` içindedir:

```bash
.venv/bin/python -m research.lab.search
```

Bu komut birkaç dakika sürebilir. İncelenmiş deneyin kopyasını ayrı bir
`HYDRA_RESEARCH_DIR` altında kullanın. Yeni veri/parametre deneyi için ayrı bir
dizin ve yeni sözleşme oluşturun; eski sonuçları yeni kazananlarla değiştirmeyin.
2025/2026 sonuçları kronolojik tarihsel test/diagnostiktir; dokunulmamış gerçek
ileri örneklem sayılmaz. OHLCV testi cüzdan veya CVD hipotezini test etmez.

## Salt okunur iki toplayıcı

Tek çevrim:

```bash
.venv/bin/python -m research.lab.okx_shadow --once
.venv/bin/python -m research.lab.hyperliquid_shadow --once
```

Sürekli gözlem için `--until` ile saat dilimi içeren, en çok 24 saat sonraki
bitiş zamanı gerekir. Örneğin 29 Ekim'de başlayacak çalışma için:

```bash
.venv/bin/python -m research.lab.okx_shadow --interval 30 --until '2026-10-30T05:00:00+00:00'
.venv/bin/python -m research.lab.hyperliquid_shadow --interval 120 --until '2026-10-30T05:00:00+00:00'
```

İki komut ayrı yönetilen terminallerde çalışır; örnekteki duruş 30 Ekim 08:00
Türkiye saati. Tarihi geçmiş bütçe reddedilir. `--once` ve `--until` birlikte
kullanılmaz. PID kilidi aynı worker'ın ikinci kopyasını engeller. Ortamın
yayımlanması veya kapanması sonrasında süreçlerin yaşayacağı varsayılmaz.

OKX kayıtları `data/shadow.sqlite3`, Hyperliquid kayıtları
`hyperliquid/observations.sqlite3` altındadır. Göreli yollar çalışma dizinine
göredir. Journal, log ve ham veri Git'e eklenmez. Varsayılan dizin `.gitignore`
ile dışarıda tutulur; farklı çalışma dizininin arşiv sorumluluğu kullanıcıdadır.

Hazır saymadan önce şunları inceleyin:

- `shadow-status.json`: yeni gözlem zamanı, gerçek trade-ID örtüşmesi ve errors.
- `hyperliquid/status.json`: başarılı hesap sayısı, fill kesilme uyarısı ve errors.
- İlk 15 dakikadaki inventory change eksik olabilir; eksik veri sıfır değildir.
- Eski veri alındı diye geçmiş receipt zamanı değiştirilmez. Yeniden başlama
  boşluğu coverage kaydında kalır; ilk geçmiş tahmin sonradan düzeltilmez.
- Hata durumu status dosyasına yazılır; son başarılı yanıt tazeymiş gibi sunulmaz.

OKX/EMA ve Hyperliquid özellikleri bağımsız araştırma kayıtlarıdır. Bu modüllerde
hesap imzası, private key, emir veya dışarı bildirim çağrısı yoktur. Yeni cüzdan
özellikleri üretim karar motoruna bağlı değildir.

## Tek rapor ve dashboard

29 Ekim ek çalışmasında, ilk altı hesaptan farklı 16 hesap ayrıca donduruldu.
Sonuçlar [devam raporundadır](CONTINUATION_20261029_TR.md). Aynı eski örneklem
değiştirilmez; yeni journal ayrı dizine yazılır. Mevcut cohort dosyalarını ezmeden:

```bash
mkdir -p data/research-wide/hyperliquid
cp -n docs/research/evidence/continued-20261029/additional-frozen-cohort.json data/research-wide/hyperliquid/large-wallet-cohort.json
HYDRA_RESEARCH_DIR="$PWD/data/research-wide" .venv/bin/python -m research.lab.hyperliquid_shadow --once --interval 240
HYDRA_RESEARCH_DIR="$PWD/data/research-wide" .venv/bin/python -m research.lab.hyperliquid_shadow --interval 240 --until '2026-10-30T05:00:00+00:00'
```

En çok 24 hesap desteklenir; 12'den fazlasında en az 240 saniye aralık zorunludur.
Yeni keşif gerekiyorsa `python -m research.lab.discover_cohort --out <yeni-dizin>`
kullanılır. Keşif sözleşmesi ilk API çağrısından önce yazılır; aynı dizinde yeniden
seçim reddedilir. Eski cohort PnL veya sonraki sonuçla değiştirilmez.

```bash
.venv/bin/python -m research.lab.breadth_report
```

Bu ek açıklayıcı kontrol toplam BTC miktar değişimini, eşit hesap yönünü ve her
hesabı sırayla çıkarınca korunabilen yönü karşılaştırır. Geçmişe uygulanan sonuçları
prospective kanıt diye etiketlemez. Sıfır hesap dengesi abstain sayılır.

```bash
.venv/bin/python -m research.lab.report
```

Bu komut network kullanmadan mevcut journal'lardan `reports/summary.json` ve
`reports/SUMMARY_TR.md` üretir. Eksik journal yaratılmaz; eksik girdiler açıkça
işaretlenir. Farklı dosyalar `--okx-db`, `--hyperliquid-db`, `--results` ve `--out`
ile seçilebilir.

Varsayılan rapor `data/research-wide/hyperliquid/observations.sqlite3` varsa ek
örneklemi de okur. Diğer yollar `--additional-hyperliquid-db` ile eklenebilir.
Bozuk ek journal ana raporun yerine başarıyla sunulmaz; ek kaynak hatası açıkça
raporlanır. Public hesap sayısı örneklemler arasında tekilleştirilir; liquidation
hacimleri körlemesine toplanmaz. Bu kayıtlar tüm piyasanın kapsamı değildir.

`GET /api/research` varsayılan raporu; henüz yoksa repo içindeki tarihli teslim
snapshot'ını sunar. `HYDRA_RESEARCH_REPORT` açıkça ayarlanırsa yalnız o dosya
kullanılır; bozuk/eksik dosyada eski teslimle sessiz fallback yapılmaz.
Arayüzdeki **Research Evidence** alanı rapor ve veri bitiş zamanını gösterir;
canlı piyasa bağlantısı durumundan bağımsızdır. Uygulama başlatma `python main.py`.

Likidasyon etüdü yalnız ilk erişim anından itibaren bilinen fill'leri gruplar.
Spot pencere sorgusu event ve receipt zamanını birlikte sınırlar. Giriş ve çıkış
fiyatı, karar sonrasındaki kaydedilmiş gerçek ticker referansıdır; mum
geçmişinden sonradan doldurulmaz. Sonuç, prospectively toplanmış veri üzerinde
**sonradan tasarlanmış açıklayıcı replay**'dir; donmuş canlı strateji testi,
gerçek fill veya fonlama/maliyet sonrası portföy getirisi değildir.

## Tamamlanan veri düzeltmeleri ve sınırlar

Cache yaşının korunması, ortak ham verinin değişmemesi, kritik fiyat/mum
kalite kapıları, kapanmış saatlik mumlar, ETF Total/tarih/pending ayrımı,
dashboard abstention ve Hyperliquid OI/funding birimleri düzeltildi.

29 Ekim ek düzeltmeleri: OKX sabit işlem sayfalaması artık gerçek zaman penceresi
CVD gibi adlandırılmaz; açık örneklem alanları ve kontrat→BTC birimi kullanılır.
Bu kaynaktan legacy CVD engine alanları üretilmez. Zamanı ve coverage'i doğrulanan
spot-flow araştırması ayrıca devam eder. Faiz değişimi yüzde puan, fiyat değişimi
yüzde kullanır. 24 borsa barı yerine saatli referans seçilir. Macro kaynak tarihi
cache'de korunur. Yan yana iki poll günlük değişim sayılmaz; BTC/SPX korelasyonu
ortak timestamp'lerde hesaplanır. FRED trade-weighted dolar serisi DXY yerine
sessizce kullanılmaz. Günlük FRED faiz gözlemlerinden FOMC toplantı tarihi türetilmesi kaldırıldı. Hardcoded takvim doğrulanmış resmî kaynak diye sunulmaz; eski tahmini macro countdown alanları üretilmez.

Eski ML simülatöründe label/fiyat eşleme, imputation, OOS çifte sayımı ve
portföy muhasebesi doğrulanmış değildir; ML kapalı tutulur. Bybit erişiminde
bölgesel HTTP 403 gözlendi. Takvim ve üçüncü taraf verilerin her alanına genel
doğruluk onayı verilmedi. Sıfır/az işlem veya çok sayıda fill, bağımsız alfa
örneklemi sayılmaz. İzlenen hesapların başka yerlerdeki hedge'leri bilinmiyor.

## TWAP forward deneyi

31 Ekim'de eklenen [TWAP forward deneyi](FORWARD_20261031_TR.md), emir göndermeyen
ayrı bir worker ve değiştirilemeyen karar günlüğü kullanır. Hipotez, kaynak
kodunun hash'leri, maliyetler ve 22 hesap ilk ağ çağrısından önce dondurulur.
Canlı deney dizininde protokol veya çekirdek uygulama değiştirilmez; farklı
tanım için yeni deney gerekir. İzleme ekranı worker başlatmaz veya sonuç üretmez.
Kurulum, duruş bütçesi, çevrimdışı yeniden üretim ve kaydedilmiş kanıt arşivi
bu yeni raporda açıklanır.

## Rakip erişimi ve yayın durumu

[Rakip analizi](COMPETITORS_TR.md) Nansen'in resmî açık referansları ile bağımsız
tracker README'lerinden doğrulananları ayırır. Pain, resmî HypurrScan arayüzü ve
bazı ticari siteler proxy 403 ile engellendi. Alan adları ortam ağ taslağına
eklendi; taslağı kaydetmek çalışan ortam erişimini veya Railway sürümünü
yayımlamaz. Ortam ayarı gerçekten uygulandıktan sonra bu sitelerin arayüz,
kapsam, gecikme ve fiyat incelemesi tamamlanabilir. Bu erişim olmadan inceleme
yapılmış gibi rapor üretilmez.

Kodun yerel doğrulaması, uzaktaki CI çalışması veya Railway deploy kanıtı
değildir. 29 Ekim kod teslimi `74bb83c` ile GitHub main'e gönderildi ve CI
başarılı oldu. Sonraki sürümün bilgisi dağıtım kılavuzundadır. Railway üzerinde
`/api/version` ve yeni API'ler görülmeden yayın doğrulanmış sayılmaz.
