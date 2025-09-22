# OpenCV ile Özellik Tespiti ve Eşleştirme

Bu proje, **OpenCV** kullanarak görüntüler üzerinde **özellik (feature) tespiti ve eşleştirme** yöntemlerini göstermektedir.  
Farklı algoritmalar kullanılarak döndürülmüş ve yeniden boyutlandırılmış görüntüler üzerinde köşe ve öznitelik eşleştirmeleri yapılır.

## Uygulanan Yöntemler
- **Harris Corner Detection** → Görüntüde köşe noktalarını bulur.  
- **SIFT (Scale-Invariant Feature Transform)** → Ölçek ve döndürmeye dayanıklı ayırt edici öznitelikler çıkarır.  
- **ORB (Oriented FAST and Rotated BRIEF)** → Gerçek zamanlı uygulamalar için hızlı ikili (binary) öznitelik çıkarımı yapar.  
- **BFMatcher (Brute Force Matcher)** → Tanımlayıcıları (descriptors) doğrudan L2 veya Hamming mesafesi ile eşleştirir.  
- **FLANN (Fast Library for Approximate Nearest Neighbors)** → KD-Tree (SIFT) veya LSH (ORB) kullanarak hızlı eşleştirme yapar.  
- **Ratio Test (Lowe’s Test)** → Güvenilir olmayan eşleşmeleri filtreler.  Güvenilir olanlar (m) good_matches_sift listesinde tutulur

##  Nasıl Çalışır
1. Girdi görüntüsü yüklenir.  
2. Görüntü döndürülür ve yeniden boyutlandırılır.  
3. **Harris Corner Detection** ile köşe noktaları bulunur.  
4. **SIFT** ve **ORB** ile öznitelikler çıkarılır.  
5. Öznitelikler şu yöntemlerle eşleştirilir:  
   - **BFMatcher (Brute Force)**  
   - **FLANN (Ratio Test ile)**  
6. Sonuçlar görselleştirilerek karşılaştırılır.  

## 📦 Gereksinimler
Aşağıdaki kütüphanelerin yüklü olması gerekir:
```bash
pip install opencv-python matplotlib numpy
# feature_detection_matching
