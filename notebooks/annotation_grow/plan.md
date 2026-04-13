**Step 1：驗證 annotation component 的品質**

在做任何連接之前，先確認你的 annotation component 本身夠好。檢查：每個 component 對應幾條 GT 纖維（理想是 1:1），有沒有嚴重的合併或碎片化。跑一下 component 數量 vs GT 纖維數量的對比，以及每個 component 跟 GT 的 overlap 分析。如果 component 本身就把不同纖維混在一起了，後面怎麼連都會有問題。

**產出**：component purity 統計，確認 component 品質足夠作為起點。

---

**Step 2：Multi-source Dijkstra 基礎實現**

在背景去除後的影像上建 cost map，所有 annotation component 的像素同時作為源頭（代價 = 0），跑一次 multi-source Dijkstra。每個像素記錄兩個值：所屬的 component ID、到達代價。

先不做任何連接，**純粹可視化**：把 Voronoi 分區畫出來，看看分區邊界是否自然地落在纖維之間的暗區。如果邊界位置合理，說明 cost map 的品質足夠引導擴張。

**產出**：Voronoi 分區圖，目視確認合理性。

---

**Step 3：加入停止條件**

沒有停止條件的 Dijkstra 會淹滿整張圖，大量背景像素也會被分配到某個 component。加入 adaptive stopping：每個 component 計算自己像素的代價統計量（mean, std），當擴張前沿的代價超過 `mean + k × std` 時停止。

先用一張圖試幾個 k 值（比如 2, 3, 5），看擴張範圍：太小會連不到鄰近 component，太大會漫進背景。

**產出**：不同 k 值下的擴張範圍可視化，選定合理的 k。

---

**Step 4：提取 component 間的候選連接**

在 Step 3 的擴張結果上，找出相鄰像素屬於不同 component 的位置——這些就是候選連接點。對每對 component，取代價最低的連接點作為最優候選。

統計一下：有多少對 component 被連接了？這些連接跟 GT 比較，precision 和 recall 各多少？也就是說，GT 中屬於同一條纖維但在不同 component 的，有多少被正確連接了？GT 中屬於不同纖維的，有多少被錯誤連接了？

**產出**：inter-component 連接的 precision / recall。

---

**Step 5：Component-level 圖建構與 MST**

把每個 component 當成一個超節點，Step 4 的候選連接作為邊（權重 = 連接代價），跑 MST 或直接用代價閾值做 pruning。

在這一步你也可以加回幾何特徵——每個 component 的端點方向現在是從整個 component 的結構估計的，比之前穩定得多。

**產出**：最終重建圖的 AHD 和 ClDice，跟之前的純 cost MST 和 DBSCAN 方法做比較。

---

**Step 6：多樣本驗證**

以上都在單張圖上做。確認方向正確後，擴展到所有樣本，看結果是否穩定。
