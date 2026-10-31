# Canlı izleme sürümü — 30 Ekim 2026

## GitHub üzerinden otomatik yayın — 31 Ekim

`.github/workflows/deploy-railway.yml`, `main` dalındaki **Tests** akışı başarılı
olunca aynı commit'i yayınlar. Fork/PR testleri dağıtım başlatmaz. Manuel Actions
çalıştırması yalnız main içindir ve önce testleri çalıştırır. GitHub production
ortamının mevcut koruma/inceleme kuralları korunur.

Akış repository veya GitHub `production` environment secrets içindeki
`RAILWAY_TOKEN` (proje/ortam token'ı) ya da `RAILWAY_API_TOKEN` (hesap/workspace
token'ı) ile çalışır. Token yoksa kimlik doğrulama adımı açık hata verir;
başarılı veya yayında gibi göstermez. Token değerleri log/artifact'a yazılmaz.
`RAILWAY_PROJECT_ID`, `RAILWAY_ENVIRONMENT_ID`, `RAILWAY_SERVICE_ID` isteğe bağlı
Variables veya Secrets olarak kapsamı daraltır; proje token'ında genelde gerekmez.

Yayıncı yalnız `web-production-57fc0.up.railway.app` alan adını taşıyan mevcut
servisi seçer. Eşleşme sıfır veya birden fazlaysa durur. Yeni Railway hesabı,
projesi veya servisi oluşturmaz; production değişkenlerini ve diğer servisleri
değiştirmez. Railway CLI 5.71.0 resmi binary'si SHA-256 ile sabitlenmiştir.

`git archive` ile yalnız test edilmiş commit paketlenir; checkout token'ı,
`.git`, yerel config ve ham journal'lar taşınmaz. Eski commit sonradan main'i
ezemez. Paketin `release.json` dosyasına gerçek Git SHA yazılır; bu build stamp,
Railway'nin önceki kaynaktan kalabilecek Git metadata'sından önceliklidir.

Upload başlatılması yayın başarısı sayılmaz. Yayıncı Railway durumunu ve public
API'leri kontrol eder: doğru release/SHA, tamamlanan döngü, taze gerçek fiyat,
22/22 cüzdan, stale olmayan snapshot, başarılı journal, maliyet API'si ve kapalı
ML/emir durumu. Sonuç GitHub Actions `railway-release-verification` artifact'ına
yazılır. Native Railway GitHub autodeploy bağlantısı olmasa da aynı repo
üzerindeki bu akış, mevcut Railway token'ıyla upload yapabilir.

Yerel salt okunur erişim kontrolü:

```bash
python scripts/railway_deploy.py --check-access
```

Bu komut token'ı kullanıcıdan okumaz veya oluşturmaz; mevcut güvenli ortam
binding'ini kullanır. Deployer kimlik doğrulamadan önce login/signup veya yeni
proje sihirbazı başlatmaz. Kaynak uygulama aynı Dockerfile ile çalışır.

Sürüm kimliği: `hydra-research-20261030.1`. Bu paket mevcut HYDRA uygulamasına
22 public hesabın canlı BTC pozisyon görünümünü, kaynak kataloğunu ve veri
bütünlüğü düzeltmelerini ekler. Cüzdanlar iki donmuş örneklem olarak kalır.
Hesap sahiplerinin bağımsızlığı ve dış hedge'leri bilinmiyor. Tahmini liquidation
seviyesi ile gerçekleşmiş liquidation fill'i farklı verilerdir.

Wallet monitor varsayılan olarak uygulamayla başlar; yaklaşık dört dakikada bir
public `/info` verisi alır. Yön motorlarına, emir API'sine veya bildirimlere
bağlı değildir. İlk yaklaşık 15 dakikada miktar değişimi unknown kalır.
Hatalı/eski hesap sıfır pozisyon sayılmaz. Kaynak hatası, cache yaşı, eksik
hesap sayısı ve journal yazma hatası API'de görünür.

## Çalıştırma

```bash
docker build -t hydra .
docker run --rm -p 8080:8080 -v hydra-data:/app/data hydra
```

`Dockerfile` Python tabanı digest ile, paketler `constraints.txt` ile sabittir.
`.dockerignore` ham data, log, `.git`, `.env`, yerel config ve credential
klasörlerini imajdan çıkarır. Kod/kanıt dosyaları içeridedir. Konteyner root
olmayan kullanıcıyla çalışır. Yönetilen proxy arkasındaki özel build ortamında
public CA bundle, isteğe bağlı BuildKit `cloud_ca` secret mount'u ile verilebilir;
TLS ve Debian paket imzaları doğrulanmaya devam eder.

Railway root dizindeki `Dockerfile`'ı otomatik algılar; uygulama Railway'nin
`PORT` değişkenini dinler. Yeni bir proje/servis oluşturmak gerekli değildir.
Varsayılan YAML template'i ve public veriler, API hesabı olmadan çalışır.
Opsiyonel sağlayıcılar mevcut Railway Variables üzerinden yapılandırılır.
Bare Python `.env` dosyasını otomatik yüklemez; değişkenler export edilmeli veya
Docker `--env-file` kullanılmalıdır.

`HYDRA_WALLET_MONITOR_ENABLED=false` yalnız canlı wallet monitor'ı kapatır.
Bu sürümde `ml.enabled: false` korunmalıdır. Maliyetlere dayanıklı alfa kabul
ölçütlerini geçen strateji bulunmadı; gerçek para/emir gönderimi yoktur.
19 araçlık katalog dış sağlayıcılara yapılmış entegrasyon anlamına gelmez;
servis denemesi yapılmamış kayıtlar API ve arayüzde böyle etiketlenir.

## Sağlık ve sürüm kontrolü

- `/api/version`: sürüm kimliği ve varsa Railway Git commit SHA.
- `/api/health`: process liveness ve tamamlanan karar döngüsü sayısı.
- `/api/status`, `/api/signal`, `/api/decision`: gerçek kaynak fiyatı, kalite,
  eşleşen karar durumu ve puanı. Sadece portun açık olması veri readiness değildir.
- `/api/wallets`: canlı örneklem; istenen/başarılı hesap sayısı, observation zamanı,
  cache stale durumu, cohort yoğunlaşması ve BTC pozisyonları.
- `/api/research`: tarihli deney raporu; deployment sırasında bundled snapshot
  kullanılabilir. Canlı wallet verisi ayrıca `/api/wallets` üzerinden gelir.
- `/api/research/tools`: 16 ürün + 3 bağımsız araç, kaynak incelemesi etiketleri.
- `/api/research/costs`: kayıtlı ilk örneklem fiyat hareketleri için varsayımsal
  maliyet senaryosu. `fee_per_side_bps=8`, `slippage_per_side_bps=4`,
  `funding_allowance_bps=0` varsayılanları 24 bp gidiş-geliş eşik oluşturur.
  Her giriş 0–100 bp aralığındadır; 1 bp = %0,01. Kullanıcı girdileri istek
  bazındadır; ortak uygulama ayarını, örneklemi veya strateji kararını değiştirmez.
  Pozitif sonuç, gerçekleşmiş net getiri veya alfa onayı değildir. Funding
  allowance her ufka uygulanır; kaynakta ölçülmüş tahakkuk değildir.

Yayın doğrulamasında `release_id`, yeni API'ler, kaynak zamanları ve gerçek
wallet okumaları birlikte kontrol edilmelidir. HTTP 200 tek başına alfa,
veri eksiksizliği veya gerçek fill kanıtı değildir.

## Veri kalıcılığı

Wallet ham yanıtları ve snapshot'lar `data/wallet-monitor.sqlite3` içine yazılır.
Railway Volume bağlıysa `RAILWAY_VOLUME_MOUNT_PATH` kökü kullanılır. Volume yoksa
container disk'i redeploy'da sıfırlanabilir; API `storage=local_disk` gösterir.
Uzun dönem araştırma için kalıcı volume ve bağımsız yedek gerekir. Yeniden
başlangıçtaki pozisyonlar yeni açılmış işlem diye etiketlenmez.

Cloud araştırma worker'ları ayrı süreçlerdir; Railway deploy'u onları oraya
taşımaz. 16 hesaplık collector için 240s poll'a karşı sabit 180s fill lookback
uyumsuzluğu düzeltildi. Yeni lookback `max(180, interval+120)`; istenmeyen aralıklar
`fill_window_gap` hatasıyla görünür ve 2.000 satır kesilmesi ayrıca korunur.
Geçmişte kaçan olaylar eksiksizmiş gibi yeniden etiketlenmez veya receipt zamanı
geri alınmaz. Bu düzeltme gelecekteki toplamayı iyileştirir; geçmişte tam piyasa
kapsamı varmış gibi sonuç çıkarılmaz.

Railway'nin [Dockerfile rehberi](https://github.com/railwayapp/docs/blob/main/content/docs/builds/dockerfiles.md)
ve [healthcheck rehberi](https://github.com/railwayapp/docs/blob/main/content/docs/deployments/healthchecks.md)
bu dağıtım yolu için incelendi.

## 30 Ekim zamanlama düzeltmesi

16 hesaplık araştırma toplayıcısında 240 saniyelik bekleme, API çevriminden sonra
başlıyordu. Yaklaşık 32 saniye kaynak süresiyle gözlem aralığı 4,53 dakikaya
çıkıyor, üçüncü eski snapshot 14 dakikadan genç ve dördüncüsü 18 dakikadan yaşlı
kalabiliyordu. 289 eski kaydın 247'sinde geçerli fark penceresi yoktu.

Yeni zamanlama çevrimin başlangıcından itibaren sabit aralık kullanır. Kaynak
işi bir periyodu aşarsa geçmiş slotlar atlanır; art arda telafi istekleri
gönderilmez. 14–18 dakika sınırı, donmuş cüzdanlar ve sonuçlara bakmayan seçim
kuralları korunur. Eski satırlar sonradan tamamlanmaz. Özellik ve health
kaydında toplama protokolü, hedef aralık ve `inventory_window_status` görünür.
`warmup`, `missing_reference`, `accounts_unavailable` sıfır işlem sayılmaz.

`/api/research` örneklem başına mevcut/eksik fark pencerelerini gösterir.
Bu sayılar tüm tarihli raporu kapsar; yeni zamanlama eski %85 eksikliği ortadan
kaldırmış gibi sunulmaz. Son canlı pencere ayrıca `/api/wallets` içinde görünür.
Maliyet senaryoları Cost Scenarios alanında ayrı, varsayımsal ve salt okunurdur.

## Native TWAP veri denemesi

`python -m research.lab.twap_probe` bir kez çalışır ve mevcut 22 donmuş public
hesabı native `userTwapSliceFills` üzerinden okur. Kendi benzersiz çıktı dizinine
ham yanıt, receipt zamanı, SHA-256 ve BTC alt kümesi yazar; önceki dizini ezmez.
İmzalama, ödeme, yeni adres seçimi veya emir gönderimi yoktur. Public query
whitelist'ine yalnız bu dokümante read türü eklendi; emir türleri reddedilir.

30 Ekim ilk gerçek denemede 22/22 yanıt başarılı, iki hesapta BTC TWAP fill'leri
vardı. İki yanıt 2.000 satır sınırındaydı; eksiksiz tarih veya piyasa kapsamı
iddiası yoktur. Geçmiş fill receipt anında yeni gerçekleşmiş işlem diye
etiketlenmez. Bu endpoint parent programın kalan miktarını ve iptal durumunu
vermez. Kanıtlar `docs/research/evidence/twap-20261030/` altındadır.
