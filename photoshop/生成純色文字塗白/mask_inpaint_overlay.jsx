/*
獨立 Photoshop JSX 腳本。

用途：
  選擇一張原圖與一張對應的文字 mask 圖，在原圖旁邊生成
 「原圖檔名_inpaint_overlay.psd」。

輸出：
  圖層：
    - inpainted: 透明覆蓋層，只包含自動填補成功的文字區域。
                 腳本會在四個角落各加 1px 白點，用來撐滿圖層 bounds。
    - bg:        原圖像素。

  Channels：
    - TEXT_CHANNEL: 保留由 mask 圖轉成的文字通道，統一為「白色=文字」。
    - OTHER_CHANNEL: 只有存在「非純色背景 / 不可靠」區域時才會保留。
                     這些區域不會填到 inpainted，而是寫入 OTHER_CHANNEL
                     方便後續人工檢查或交給其他 inpaint 流程。

原理：
  1. mask 可以是黑底白字，也可以是白底黑字；腳本會把較少出現的
     極端色視為文字前景，必要時反相，得到 TEXT_CHANNEL。
  2. 從 TEXT_CHANNEL 生成選區，再轉成工作路徑，取得文字形狀的
     局部邊界框，並把相鄰邊界框合併成文字塊。
  3. 對每個文字塊：
       repair area = 文字區域向外擴張 REPAIR_EXPAND_PX，
                     用來覆蓋抗鋸齒和殘留邊。
       sample ring = repair area 再向外擴張 SAMPLE_RING_PX，
                     並扣掉 repair area 和所有文字 mask。
  4. 在原圖的 sample ring 上做 RGB histogram 分析：
       - 如果背景像純色，取主色填入 inpainted 的 repair area。
       - 如果背景不像純色，跳過填色，將 repair area 寫入 OTHER_CHANNEL。

純色判斷參數：
  SOLID_P90_P10_MAX：
    對 sample ring 的 R/G/B histogram 分別計算 P90 - P10。
    數值越大，代表顏色越分散，越可能是漸層、網點、線稿或邊框污染。
    這個值越小越嚴格；越大越寬鬆。

  SOLID_PEAK_RATIO_MIN：
    對 sample ring 的 R/G/B histogram 分別找出最高峰佔比。
    主色佔比越高，越像純色背景。
    這個值越高越嚴格；越低越寬鬆。

建議調整：
  如果太多純白/純黑泡泡被放進 OTHER_CHANNEL：
    提高 SOLID_P90_P10_MAX，例如 25；
    降低 SOLID_PEAK_RATIO_MIN，例如 0.45。

  如果網點、漸層、線稿背景被錯誤填色：
    降低 SOLID_P90_P10_MAX，例如 12；
    提高 SOLID_PEAK_RATIO_MIN，例如 0.65。
*/

#target photoshop

(function () {
    var REPAIR_EXPAND_PX = 4;      // 擴張文字修補區，用來覆蓋抗鋸齒邊緣。
    var SAMPLE_RING_PX = 14;       // 在修補區外側取樣背景色的距離。
    var GROUP_MERGE_PX = 16;       // 合併相鄰字形/path 邊界框，形成局部文字塊。
    var BLOCK_PADDING_PX = 8;      // 區塊與 mask 相交前額外放大的範圍。
    var PATH_TOLERANCE_PX = 2;     // 將選區轉成工作路徑時的容差。
    var MIN_BOX_SIZE_PX = 2;
    var SOLID_P90_P10_MAX = 18;     // P90-P10 越大，越可能是漸層、網點或線稿。
    var SOLID_PEAK_RATIO_MIN = 0.55;// 取樣環中的主色必須佔足夠比例。

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        var originalFile = File.openDialog("Select original image", "Images:*.psd;*.png;*.jpg;*.jpeg;*.tif;*.tiff");
        if (!originalFile) return;

        var maskFile = File.openDialog("Select text mask image", "Images:*.psd;*.png;*.jpg;*.jpeg;*.tif;*.tiff");
        if (!maskFile) return;

        var outFile = new File(originalFile.path + "/" + stripExtension(originalFile.name) + "_inpaint_overlay.psd");

        var outDoc = createOutputDocument(originalFile);
        var bgLayer = outDoc.artLayers.getByName("bg");
        var inpaintLayer = outDoc.artLayers.add();
        inpaintLayer.name = "inpainted";
        inpaintLayer.move(bgLayer, ElementPlacement.PLACEBEFORE);

        var textChannel = pasteMaskIntoAlpha(outDoc, maskFile);

        var boxes = makeBoxesFromMaskSelection(outDoc, textChannel);
        boxes = mergeBoxes(boxes, GROUP_MERGE_PX, outDoc.width.as("px"), outDoc.height.as("px"));

        if (boxes.length === 0) {
            alert("No text region found in the mask.");
            cleanupChannels(outDoc, [textChannel]);
            return;
        }

        var repairChannel = outDoc.channels.add();
        repairChannel.name = "__repair_area";
        var otherChannel = outDoc.channels.add();
        otherChannel.name = "OTHER_CHANNEL";
        var hasOtherRegion = false;

        for (var i = 0; i < boxes.length; i++) {
            if (processBlock(outDoc, bgLayer, inpaintLayer, textChannel, repairChannel, otherChannel, boxes[i])) {
                hasOtherRegion = true;
            }
        }
        addBoundsAnchorPixels(outDoc, inpaintLayer);

        cleanupChannels(outDoc, [repairChannel]);
        if (!hasOtherRegion) {
            cleanupChannels(outDoc, [otherChannel]);
        }
        outDoc.selection.deselect();

        var saveOptions = new PhotoshopSaveOptions();
        saveOptions.layers = true;
        outDoc.saveAs(outFile, saveOptions, true, Extension.LOWERCASE);

        alert("Done.\nBlocks processed: " + boxes.length + "\nSaved: " + outFile.fsName);
    } catch (e) {
        alert("Mask inpaint overlay failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function createOutputDocument(originalFile) {
        var srcDoc = app.open(originalFile);
        var docName = stripExtension(originalFile.name) + "_inpaint_overlay";

        var outDoc = srcDoc.duplicate(docName, true);
        srcDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = outDoc;
        if (outDoc.mode !== DocumentMode.RGB) {
            outDoc.changeMode(ChangeMode.RGB);
        }
        outDoc.activeLayer = outDoc.layers[0];
        outDoc.activeLayer.name = "bg";

        return outDoc;
    }

    function pasteMaskIntoAlpha(outDoc, maskFile) {
        var widthPx = outDoc.width.as("px");
        var heightPx = outDoc.height.as("px");

        var maskDoc = app.open(maskFile);
        if (Math.round(maskDoc.width.as("px")) !== Math.round(widthPx) ||
            Math.round(maskDoc.height.as("px")) !== Math.round(heightPx)) {
            maskDoc.resizeImage(UnitValue(widthPx, "px"), UnitValue(heightPx, "px"), undefined, ResampleMethod.NEARESTNEIGHBOR);
        }

        var invert = shouldInvertMask(maskDoc);

        maskDoc.selection.selectAll();
        maskDoc.selection.copy();

        app.activeDocument = outDoc;
        var alpha = outDoc.channels.add();
        alpha.name = "TEXT_CHANNEL";
        outDoc.activeChannels = [alpha];
        outDoc.selection.selectAll();
        outDoc.paste();
        outDoc.selection.deselect();

        if (invert) {
            outDoc.activeChannels = [alpha];
            invertActiveChannel();
        }

        setRGBChannels(outDoc);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);

        return alpha;
    }

    function shouldInvertMask(maskDoc) {
        var hist = maskDoc.channels[0].histogram;
        var dark = 0;
        var light = 0;

        for (var i = 0; i <= 31; i++) dark += hist[i];
        for (var j = 224; j <= 255; j++) light += hist[j];

        // 載入 alpha 通道時會選中亮色區域；如果較少的前景是淺色 mask
        // 上的深色文字，就需要反相，統一成「白色=文字」。
        return dark < light;
    }

    function makeBoxesFromMaskSelection(doc, textChannel) {
        setRGBChannels(doc);
        doc.selection.load(textChannel, SelectionType.REPLACE);

        if (!hasSelection(doc)) {
            return [];
        }

        doc.selection.makeWorkPath(PATH_TOLERANCE_PX);
        var pathItem = doc.pathItems[doc.pathItems.length - 1];
        var boxes = [];

        for (var i = 0; i < pathItem.subPathItems.length; i++) {
            var box = boxFromSubPath(pathItem.subPathItems[i]);
            if (box && (box.r - box.l) >= MIN_BOX_SIZE_PX && (box.b - box.t) >= MIN_BOX_SIZE_PX) {
                boxes.push(box);
            }
        }

        pathItem.remove();
        doc.selection.deselect();
        return boxes;
    }

    function boxFromSubPath(subPath) {
        var minX = 99999999;
        var minY = 99999999;
        var maxX = -99999999;
        var maxY = -99999999;

        for (var i = 0; i < subPath.pathPoints.length; i++) {
            var anchor = subPath.pathPoints[i].anchor;
            var x = coordToPx(anchor[0]);
            var y = coordToPx(anchor[1]);

            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
        }

        if (minX === 99999999) return null;
        return { l: minX, t: minY, r: maxX, b: maxY };
    }

    function processBlock(doc, bgLayer, inpaintLayer, textChannel, repairChannel, otherChannel, block) {
        var docWidth = doc.width.as("px");
        var docHeight = doc.height.as("px");
        var blockRect = clampBox(expandBox(block, BLOCK_PADDING_PX), docWidth, docHeight);

        setRGBChannels(doc);
        selectRect(doc, blockRect);
        doc.selection.load(textChannel, SelectionType.INTERSECT);

        if (!hasSelection(doc)) {
            doc.selection.deselect();
            return false;
        }

        doc.selection.expand(REPAIR_EXPAND_PX);
        doc.selection.store(repairChannel, SelectionType.REPLACE);

        doc.selection.load(repairChannel, SelectionType.REPLACE);
        doc.selection.expand(SAMPLE_RING_PX);
        doc.selection.load(repairChannel, SelectionType.DIMINISH);
        doc.selection.load(textChannel, SelectionType.DIMINISH);

        if (!hasSelection(doc)) {
            doc.selection.load(repairChannel, SelectionType.REPLACE);
            doc.selection.deselect();
            return false;
        }

        inpaintLayer.visible = false;
        doc.activeLayer = bgLayer;
        var quality = getSelectionSolidQuality(doc);
        var fillColor = getSelectionDominantColor(doc);
        inpaintLayer.visible = true;

        if (!quality.isSolid) {
            doc.selection.load(repairChannel, SelectionType.REPLACE);
            doc.selection.store(otherChannel, SelectionType.EXTEND);
            doc.selection.deselect();
            return true;
        }

        doc.activeLayer = inpaintLayer;
        doc.selection.load(repairChannel, SelectionType.REPLACE);
        doc.selection.fill(fillColor, ColorBlendMode.NORMAL, 100, false);
        doc.selection.deselect();
        return false;
    }

    function getSelectionSolidQuality(doc) {
        var r = analyzeHistogram(doc.channels[0].histogram);
        var g = analyzeHistogram(doc.channels[1].histogram);
        var b = analyzeHistogram(doc.channels[2].histogram);
        var maxSpread = Math.max(r.spread, Math.max(g.spread, b.spread));
        var minPeakRatio = Math.min(r.peakRatio, Math.min(g.peakRatio, b.peakRatio));

        return {
            isSolid: maxSpread <= SOLID_P90_P10_MAX && minPeakRatio >= SOLID_PEAK_RATIO_MIN,
            maxSpread: maxSpread,
            minPeakRatio: minPeakRatio
        };
    }

    function analyzeHistogram(hist) {
        var total = 0;
        var maxCount = 0;
        var p10 = 0;
        var p90 = 255;

        for (var i = 0; i < 256; i++) {
            total += hist[i];
            if (hist[i] > maxCount) maxCount = hist[i];
        }

        if (total <= 0) {
            return { spread: 255, peakRatio: 0 };
        }

        var lowTarget = total * 0.10;
        var highTarget = total * 0.90;
        var acc = 0;
        var gotP10 = false;
        for (var j = 0; j < 256; j++) {
            acc += hist[j];
            if (!gotP10 && acc >= lowTarget) {
                p10 = j;
                gotP10 = true;
            }
            if (acc >= highTarget) {
                p90 = j;
                break;
            }
        }

        return {
            spread: p90 - p10,
            peakRatio: maxCount / total
        };
    }

    function getSelectionDominantColor(doc) {
        function dominant(hist) {
            var idx = 0;
            var max = hist[0];
            for (var i = 1; i < hist.length; i++) {
                if (hist[i] > max) {
                    idx = i;
                    max = hist[i];
                }
            }
            return idx;
        }

        var c = new SolidColor();
        c.rgb.red = dominant(doc.channels[0].histogram);
        c.rgb.green = dominant(doc.channels[1].histogram);
        c.rgb.blue = dominant(doc.channels[2].histogram);
        return c;
    }

    function addBoundsAnchorPixels(doc, layer) {
        var w = doc.width.as("px");
        var h = doc.height.as("px");
        var white = new SolidColor();
        white.rgb.red = 255;
        white.rgb.green = 255;
        white.rgb.blue = 255;

        doc.activeLayer = layer;
        fillPixel(doc, 0, 0, white);
        fillPixel(doc, w - 1, 0, white);
        fillPixel(doc, 0, h - 1, white);
        fillPixel(doc, w - 1, h - 1, white);
        doc.selection.deselect();
    }

    function fillPixel(doc, x, y, color) {
        doc.selection.select([
            [x, y],
            [x + 1, y],
            [x + 1, y + 1],
            [x, y + 1]
        ], SelectionType.REPLACE, 0, false);
        doc.selection.fill(color, ColorBlendMode.NORMAL, 100, false);
    }

    function mergeBoxes(boxes, pad, docWidth, docHeight) {
        var merged = [];

        for (var i = 0; i < boxes.length; i++) {
            merged.push(clampBox(boxes[i], docWidth, docHeight));
        }

        var changed = true;
        while (changed) {
            changed = false;

            for (var a = 0; a < merged.length; a++) {
                for (var b = a + 1; b < merged.length; b++) {
                    if (boxesNear(merged[a], merged[b], pad)) {
                        merged[a] = unionBox(merged[a], merged[b]);
                        merged.splice(b, 1);
                        changed = true;
                        break;
                    }
                }
                if (changed) break;
            }
        }

        for (var k = 0; k < merged.length; k++) {
            merged[k] = clampBox(expandBox(merged[k], BLOCK_PADDING_PX), docWidth, docHeight);
        }

        return merged;
    }

    function boxesNear(a, b, pad) {
        return !(
            a.r + pad < b.l ||
            b.r + pad < a.l ||
            a.b + pad < b.t ||
            b.b + pad < a.t
        );
    }

    function unionBox(a, b) {
        return {
            l: Math.min(a.l, b.l),
            t: Math.min(a.t, b.t),
            r: Math.max(a.r, b.r),
            b: Math.max(a.b, b.b)
        };
    }

    function expandBox(box, pad) {
        return {
            l: box.l - pad,
            t: box.t - pad,
            r: box.r + pad,
            b: box.b + pad
        };
    }

    function clampBox(box, width, height) {
        return {
            l: Math.max(0, Math.floor(box.l)),
            t: Math.max(0, Math.floor(box.t)),
            r: Math.min(width - 1, Math.ceil(box.r)),
            b: Math.min(height - 1, Math.ceil(box.b))
        };
    }

    function selectRect(doc, box) {
        doc.selection.select([
            [box.l, box.t],
            [box.r, box.t],
            [box.r, box.b],
            [box.l, box.b]
        ], SelectionType.REPLACE, 0, false);
    }

    function hasSelection(doc) {
        try {
            var bounds = doc.selection.bounds;
            return (bounds[2].as("px") - bounds[0].as("px") > 0) &&
                   (bounds[3].as("px") - bounds[1].as("px") > 0);
        } catch (e) {
            return false;
        }
    }

    function setRGBChannels(doc) {
        doc.activeChannels = [doc.channels[0], doc.channels[1], doc.channels[2]];
    }

    function invertActiveChannel() {
        app.executeAction(app.charIDToTypeID("Invr"), undefined, DialogModes.NO);
    }

    function cleanupChannels(doc, channels) {
        setRGBChannels(doc);
        for (var i = 0; i < channels.length; i++) {
            try {
                channels[i].remove();
            } catch (e) {
            }
        }
    }

    function coordToPx(v) {
        if (v && v.as) {
            return v.as("px");
        }
        return Number(v);
    }

    function stripExtension(name) {
        return name.replace(/\.[^\.]+$/, "");
    }
})();
