# Telefonda HYDRA Spot denemesi

## Hazır olan

`feature/spot-wallet-preview-20261103` dalında `/spot/` ekranı:

- MetaMask/Rabby uygulama içi tarayıcılarında enjekte edilen cüzdan bağlantısı.
- Hyperliquid spot USDC bakiyesi ve UBTC/USDC işlem geçmişi.
- Güncel emir defteri, kişisel spot ücret oranı ve miktar üzerinden önizleme.
- Kullanıcının her emir için ayrı imzası; tek UBTC alım emri, IOC ve %0,5 kayma
  sınırı. 12–1.000 USDC tutar sınırı, komisyon payı ve 30 saniyelik önizleme süresi.
- Cüzdan/ağ değişiminde önizleme iptali; belirsiz gönderimde otomatik tekrar yok.

Özel anahtar, seed, API agent yetkisi, token allowance, para yatırma veya çekim
istenmiyor. Sunucuda emir imzalayan anahtar veya işlem endpoint'i eklenmedi.
UBTC, Bitcoin ağındaki yerel BTC değil, Unit üzerinden temsil edilen varlık.

## Doğrulama

- 238 Python testi geçti; spot route ve derlenmiş dosya erişimi dahil.
- 10 TypeScript testi geçti; token kimliği, hassas miktar/fiyat yuvarlama, komisyon
  rezervi, süresi dolan fiyat, yanlış imzalayan cüzdan ve kısmi sonuç ayrımı dahil.
- Emir özeti, resmi Hyperliquid Python SDK'sının `action_hash` fonksiyonuyla aynı
  sonucu verdi; nonce ve son geçerlilik zamanı da özete dahil.
- Chromium: 390px mobil görünüm, taşma yok, sahte sağlayıcıyla bağlantı/önizleme,
  yetersiz bakiye ve imza zinciri uyuşmazlığında gönderim engeli; tarayıcı hatası yok.
  Bu test gerçek cüzdana veya gerçek 100 USDC bakiyeye dair kanıt değildir.
- Canlı `/info` okumasında UBTC/USDC varlık kimliği 10142 doğrulandı. Tarayıcı CORS
  başlıkları ve `/exchange` OPTIONS ön kontrolü uygun. Gerçek exchange POST sayısı
  **sıfır**; gerçek telefon cüzdanında alım henüz doğrulanmadı.

Bazı cüzdanlar Hyperliquid'in `Exchange / Agent`, zincir kimliği 1337 olan imza
alanını reddeder. Böyle bir cüzdanda bağlantı ve önizleme kullanılabilir; gerçek
emir gönderilmez. Sırf bu engeli aşmak için sahte ağ veya geniş işlem yetkisi
isteyen API agent eklenmedi. WalletConnect proje kimliği yapılandırılmadı.

## Yayın durumu ve gereken son adım

[GitHub Actions derlemesi ve 10 test başarılı](https://github.com/BerkayMangal/HYDRA-PRODUCT/actions/runs/41669259036).
[Ayrı Python CI başarılı](https://github.com/BerkayMangal/HYDRA-PRODUCT/actions/runs/41669258696).

Pages yayın işi site oluştururken HTTP 403 döndürdü:
`Resource not accessible by integration`. Depoda Pages sitesi bulunamadı;
Actions'ın GITHUB_TOKEN'ı ilk siteyi açma yönetim yetkisine sahip değil.
Bu hata otomatik güvenlik onayının reddi değil, GitHub API yetki hatasıdır.

Depo sahibi için kalan adım:

1. [Settings → Pages](https://github.com/BerkayMangal/HYDRA-PRODUCT/settings/pages)
   sayfasında Build and deployment → Source: **GitHub Actions** seç.
2. Yukarıdaki Spot phone preview işinde **Re-run failed jobs** çalıştır.
3. İşteki HTTPS doğrulaması başarılı olduktan sonra çıktıda verilen `/spot/`
   adresini MetaMask/Rabby uygulamasının tarayıcısında aç.

Beklenen adres `https://berkaymangal.github.io/HYDRA-PRODUCT/spot/`;
**bu rapor yazıldığında yayınlanmış/çalışır olarak doğrulanmadı**. Railway üretim
adresi güncellenmedi. Railway yönetim anahtarı bu oturumda mevcut değil.

## Gelir modeli sorusu

Hyperliquid **Builder Code / Builder Fee**, kullanıcı tarafından onaylanan
azami ücret ve emre eklenen builder adresi/ücret alanıyla çalışıyor. **Referral
Code** ayrı bir yönlendirme programı. Sadece cüzdan bağlanması ücret doğurmuyor.
Resmi SDK örneği:
https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/examples/basic_builder_fee.py

Bu sürüm hiçbir builder ücreti veya builder ücret onayı istemiyor. Spot alım
özelindeki güncel gelir uygunluğu ve tahsil kuralları bağımsız resmi dokümanla
henüz doğrulanmadığı için alım başına gelir sözü veya gelir tahmini yapılmadı.
Bu soru sebebiyle kullanıcı hesabına referral/builder onayı gönderilmedi.
