# 31 Ekim: veri bütünlüğü ve donmuş TWAP forward deneyi

Henüz doğrulanmış alfa yok. Bu teslim, tek bir hipotezi gerçek zamanlı
gözlemler üzerinde sınamak ve her kararı daha sonra aynı kayıttan doğrulamak
için hazırlandı. Önceki 13.368 varyasyonun sonuçları yeni deneye eğitim veya
başarı kanıtı olarak aktarılmadı. Para yöneten bir portföy veya emir sistemi
başlatılmadı.

## Veriye ilişkin bulgular

[İlk denetim kaydı](evidence/forward-20261031/historical-data-audit.json) mevcut
üç journal üzerinde salt okunur olarak üretildi. Bütün kaydedilmiş ham JSON
payload'larının SHA256 değerleri yeniden hesaplandı:

| Kaynak | Kontrol edilen payload | İşlem/fill satırı | Hash uyuşmazlığı |
|---|---:|---:|---:|
| OKX | 108.327 | 8.309.786 | 0 |
| İlk 6 Hyperliquid hesabı | 10.437 | 965.200 | 0 |
| İlave 16 Hyperliquid hesabı | 7.955 | 1.622.815 | 0 |
| Toplam | 126.719 | | 0 |

Üç SQLite `quick_check` sonucu da `ok`. İşlem zamanı ilk gözlem zamanından
ileride olan satır bulunmadı. OKX fiyat × BTC miktarı ile USDT tutarı arasında
kontrol toleransını aşan fark bulunmadı; spot dönüşümü 1, perp kontratının BTC
dönüşümü 0,01. 30 Ekim'deki 22 donmuş TWAP yanıt arşivinin hash'leri de eşleşti.

Bu bulgular, kayıtların kendi içinde tutarlı olduğunu gösterir. Kaynak
borsanın bütün işlemlerini aldığımızı veya upstream verinin her alanının doğru
olduğunu kanıtlamaz. OKX'te **1.007 degraded toplama döngüsü** var. İlk hesap
grubunda **153**, ilave grupta **51** döngüde fill yanıt limiti görülmüş.
Bu boşluklar silinmedi veya sıfır hacimle doldurulmadı.

Önceki toplama aralığı düzeltmesinden sonra 14–18 dakikalık envanter referansı,
ilk grupta 356 gözlemin 349'unda, ilave grupta 179 gözlemin 173'ünde mevcut.
Bunlar takvim/snapshot kapsama sayılarıdır; bağımsız kazançlı işlem sayısı değildir.

Aynı tanılamalar artık tekrar çalıştırılabilir:

```bash
python -m research.lab.data_audit --out /tmp/hydra-data-audit.json
```

Farklı dosyalar için `--okx`, `--primary`, `--additional` kullanılır. Eksik veya
bozuk kaynak başarılı sayılmaz; eksik SQLite dosyası oluşturulmaz. Araç hiçbir
kaydı onarmaz veya değiştirmez. [Tekrarlanmış denetim çıktısı](evidence/forward-20261031/reproducible-data-audit.json)
daha sıkı eksik/değersiz zaman ve sayı kontrollerini de içerir.

## Önceden belirlenen hipotez

Son 15 dakikada izlenen hesapların gerçekleşmiş BTC TWAP dilimlerindeki net
alış miktarı pozitifse, bir sonraki 15 dakikada tamamen fonlanmış OKX BTC-USDT
spot long pozisyonunun maliyet sonrası hareketi pozitif olur mu?

[Deney tanımı](../../research/twap_forward_spec.json) ilk veri çağrısından önce
dosyaya donduruldu. Önceden dondurulmuş iki örneklemden gelen **22 public hesap**
tekilleştiriliyor; performanslarına bakıp hesap değiştirilmedi. Kontrol modeli,
önceki 15 dakikalık spot fiyat momentumu pozitif olduğunda long, diğer geçerli
durumlarda flat kalıyor. İki model aynı fiyat referanslarını kullanıyor.

Bu mekanizma, gerçekleşmiş TWAP dilimlerinin devam eden alım baskısı hakkında
bilgi taşıyabileceği fikrini sınar. Kalan ana emir büyüklüğü, iptal durumu,
hesapların ortak sahibi veya başka yerlerdeki hedge'leri bilinmiyor. İzlenen
hesaplardaki akış bütün piyasanın akışı değildir. İlk yeni dilimler yalnız bir
aktif hesaptan geldi; sonuçlarda hesap yoğunlaşması ayrıca görünür.

## Yanlış başarıyı engelleyen kurallar

- İlk hesap yanıtındaki geçmiş işlemler **bootstrap**; gelecekteki sinyallere
  dönüşmüyor. Aynı işlem tekrar gelince ilk erişim zamanı yenilenmiyor.
- Feature penceresinin ortak bitişi batch içindeki ilk istek zamanıdır. Karar
  batch tamamlandıktan sonra kaydedilir; geç gelen bilgi geçmişe taşınmaz.
- 15 dakika kesintisiz gözlem, bütün hesapların başarılı yanıtı, uygun poll
  aralığı ve boş olmayan ardışık geçmişlerde gerçek trade-ID örtüşmesi gerekir.
  Yanıt 2.000 satıra ulaşırsa geçmişin pencere başına uzanması da gerekir.
  Bu kontrol, sağlayıcının yakın geçmiş sözleşmesi altında sınırlı bir kapsama
  varsayımıdır; tam borsa geçmişi garantisi değildir.
- Her UTC çeyrek saatin ilk tamamlanan batch'i tek karar hakkını kullanır.
  Warmup veya kalite nedeniyle pas geçilen karar sonradan yeniden yazılmaz.
  Önceki eşleştirilmiş olay beklerken yeni pozisyon açılmaz.
- Giriş, karar + 1 saniyeden sonra ilk gözlenen geçerli **ask**; çıkış,
  giriş gözleminden 15 dakika sonra ilk geçerli **bid** referansıdır. Referans
  en çok 30 saniye gecikebilir; kaynak fiyat yaşı en çok 5 saniye olabilir.
  Görünen en iyi kademedeki miktar model miktarını karşılamalıdır.
- Geciken, değişen, geleceğe yazılmış veya kapsaması belirsiz kaynak verisi
  karar engelidir. Eksik giriş/çıkış ve yetersiz kademe miktarı **unknown** olur;
  sıfır getiriye veya daha sonraki uygun fiyata dönüştürülmez.

Sabit olay büyüklüğü **1.000 USDT**, varsayılan ücret **8 bp/yan**, kayma
**4 bp/yan**. Alış-satış farkı gözlenen fiyatlardan ayrıca gelir. Ücretler hem
girişte hem çıkışta uygulanır. Model long/flat spot kullandığından borçlanma ve
perp fonlaması yoktur. Fiyat referansları gerçek fill garantisi değildir.

Bu çalışma sabit tutarlı, çakışmayan olay karşılaştırmasıdır; bileşik portföy
NAV'ı, Sharpe veya gerçek hesap kârı olarak sunulmaz. Örneklenmiş MFE/MAE,
10 saniyelik kayıtlardan hesaplanır ve tick bazında en yüksek/en düşük hareket
anlamına gelmez. Kontrolün geçerli flat kaldığı tamamlanmış karşılaştırmalarda
flat getirisi yalnız eşleştirilmiş fark hesabında sıfır kabul edilir; flat
kararı kazançlı işlem sayılmaz.

## Gerçek çalışma ve kayıtlar

Canlı ileri gözlem **31 Ekim 2026 10:19:15 UTC**'de başladı. Bitiş bütçesi
**1 Kasım 2026 09:00 UTC** (bu tarihlerde Londra saatiyle aynı). Süreç yalnız
bu çalışma ortamında çalışır; ortamın kapanmasından sonra yaşayacağı varsayılmaz.
Eski üç bounded worker'ın bitiş bütçeleri korunmuş, yeniden başlatılmamıştır.

Deney çekirdeğinin Git revizyonu `d703297477d022146d20419c01a55a5092383364`.
Protokol SHA256 değeri her raporda yer alır. Protokol, iki çekirdek dosyanın
hash'lerini de saklar; farklı kod veya tanımla aynı deney dizini sürdürülemez.
Sonraki arayüz ve salt okunur raporlama değişiklikleri bu çekirdeği değiştirmez.

`ledger.sqlite3`, normal uygulama yazımlarında UPDATE/DELETE/REPLACE işlemlerini
reddeden tetikleyicilerle korunur. Kaynak yanıtları, ilk erişimler, fiyatlar,
kararlar, girişler, sonuçlar ve worker duruşları eklenerek saklanır. Her karar
girdi trade-ID listesini ve hash'ini, her giriş/çıkış gerçek kayıtlı quote-ID'sini
taşır. Bu mekanizma dosyaya sınırsız erişimi olan bir kişinin veriyi değiştirmesine
karşı kriptografik bir dış zaman damgası değildir; kaydedilmiş hash arşivi ek
karşılaştırma noktası sağlar.

31 Ekim **10:36:02 UTC** dondurmasında beş batch'in hepsi 22/22 başarılı;
istek hatası ve reddedilmiş quote yok. Limitli 10 yanıt kaydı, tekrarlanan
hesap yanıtlarıdır; 10 farklı hesap anlamına gelmez. Son batch'in sınırlı
kapsama kapıları geçti. Journal'da 2.510 bootstrap, 35 yeni uygun BTC dilimi
var; ortak pencere bitişinden önce kalan 33 dilimin neti **+0,00941 BTC**, aktif
hesap sayısı **1**. Bu küçük akış, alım önerisi veya alfa sonucu değildir.

İlk iki çeyrek saat kararı warmup sırasında pas geçildi. Sonradan tamamlanan
pencere bu eski kararları değiştirmedi. Arşivde tamamlanmış model işlemi **0**,
ortalama getiri **unknown**; gerçek performans iddiası yok. Arşivdeki **211** yeni
ham yanıt/fiyat kaydının hash kontrolü geçti. Çevrimdışı yeniden üretim birebir
aynı raporu verdi. Yerel uygulama testlerinde **207 test geçti**; sentetik test
fixture'ları performans kanıtı olarak kullanılmadı. Chromium masaüstü/mobil
kontrollerinde JavaScript istisnası ve 390 px ekranda sayfa taşması görülmedi.

Yeni ve ayrı bir deney için:

```bash
python -m research.lab.twap_forward --once --out data/research/twap-forward/new-smoke
python -m research.lab.twap_forward --out data/research/twap-forward/new-trial \
  --until '2026-11-01T09:00:00+00:00' \
  --report-mirror data/research/twap-forward/latest-summary.json
```

`--until`, başlatıldığı anda gelecekte ve en çok 24 saat uzakta olmalıdır.
Örnekteki tarih geçmişse yeni tarih seçilir. Aynı deney için ikinci worker
kilitle engellenir; 1 GiB altı disk alanında toplama durur. Kaynaklar public;
hesap imzası, private key, ücretli API veya bildirim gerektirmez. Deney yalnız
veri okur ve model kayıtları yazar.

## Çevrimdışı kontrol ve ekran

`GET /api/research/forward` ve **TWAP Forward Trial** paneli yalnız kaydedilmiş
raporu okur. Sayfa açmak veri toplama başlatmaz, işlem sonuçlandırmaz. Worker
30 saniyede bir atomik rapor yayınlar. Eski rapor taze gibi gösterilmez; kaynak
okumalarının güncelliği ile 15 dakikalık sinyal penceresinin uygunluğu ayrı
alanlardır. Varsayılan rapor yoksa `not_started` döner; repo içindeki tarihli
başarı örneğine fallback yapılmaz. `HYDRA_FORWARD_REPORT` ile açık dosya yolu
seçilebilir; eksik veya bozuksa `unavailable` olur.

[Dondurulmuş kanıt](evidence/forward-20261031/manifest.json) çalışan SQLite'tan
backup API ile tutarlı olarak alınır; açık WAL dosyası körlemesine kopyalanmaz.
Arşiv ve açılmış SQLite hash'leri manifestte yer alır. Yeniden üretmek için:

```bash
mkdir -p /tmp/hydra-forward-recorded
cp docs/research/evidence/forward-20261031/protocol.json /tmp/hydra-forward-recorded/
gzip -dc docs/research/evidence/forward-20261031/ledger.sqlite3.gz > /tmp/hydra-forward-recorded/ledger.sqlite3
python -m research.lab.twap_forward --audit --out /tmp/hydra-forward-recorded
python -m research.lab.forward_report --trial /tmp/hydra-forward-recorded \
  --out /tmp/hydra-forward-reproduced.json
```

Rapor zamanı en son kaydedilmiş olaydan hesaplanır; raporu aylar sonra açmak
örneklemi yaşlandırmaz, işlem tamamlamaz veya yeni gözlem eklemez. Kaynak kodu
ileride değişirse manifestteki çekirdek revizyonuna uygun kod gerekir.
[Kaydedilmiş rapor](evidence/forward-20261031/recorded-report.json) ile aynı
çıktı elde edilmelidir. Denetim sadece mevcut referansları doğrular; henüz giriş
veya çıkış yoksa sıfır ihlal sonucu bir gerçekleşmiş işlem kanıtı değildir.

En az **100 tamamlanmış strateji işlemi**, **14 gün** ve **0 belirsiz aktif
sonuç** yalnız sonraki değerlendirmeye geçiş eşiğidir. İstatistiksel bağımlılık,
kontrole göre fark, piyasa rejimleri ve maliyet hassasiyeti ayrıca incelenmeden
alfa iddiası kurulmaz. Hiçbir eşik otomatik emir açmaz.

Değişiklikler `research/twap-forward-20261031` araştırma dalındadır. Yerel ekran
ve kayıtlı önizleme Railway yayını değildir; bu çalışma Railway sürümünü veya
`main` dalını değiştirmez.
