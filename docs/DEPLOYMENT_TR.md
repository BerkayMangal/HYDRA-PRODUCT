# Canlı izleme sürümü — 29 Ekim 2026

Sürüm kimliği: `hydra-research-20261029.1`. Bu paket mevcut HYDRA uygulamasına
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
