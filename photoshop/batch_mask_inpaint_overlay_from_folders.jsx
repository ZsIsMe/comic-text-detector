/*
Batch mask inpaint overlay generator.

Run in Photoshop:
File > Scripts > Browse... > batch_mask_inpaint_overlay_from_folders.jsx

Input:
- Image folder
- Mask folder, auto-filled as <image folder>/mask after selecting image folder

Output:
<image folder>/inpaint_overlay_psds/<image name>.psd
<image folder>/inpaint_overlay_psds/other_channel_report.txt
Optional:
<image folder>/inpaint_overlay_psds/png/<image name>.png

Each PSD contains:
- bg: original image pixels.
- bubble: transparent overlay filled only where the text background looks solid.
- TEXT_CHANNEL: normalized text mask, white means text.
- OTHER_CHANNEL: kept only when non-solid / unreliable regions exist.
*/

#target photoshop

(function () {
    app.bringToFront();

    var REPAIR_EXPAND_PX = 3;
    var SAMPLE_RING_PX = 6;
    var GROUP_MERGE_PX = 16;
    var BLOCK_PADDING_PX = 8;
    var PATH_TOLERANCE_PX = 2;
    var MIN_BOX_SIZE_PX = 2;
    var SOLID_P90_P10_MAX = 18;
    var SOLID_PEAK_RATIO_MIN = 0.55;
    var WHITE_DOMINANT_MIN = 235;
    var WHITE_PEAK_RATIO_MIN = 0.58;

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        var settings = showSettingsDialog();
        if (!settings) return;

        var imageFolder = new Folder(settings.imageFolder);
        var maskFolder = new Folder(settings.maskFolder);
        var exportPng = settings.exportPng;
        var runActionWhenOther = settings.runActionWhenOther;

        if (!imageFolder.exists) {
            alert("原图文件夹不存在：\n" + imageFolder.fsName);
            return;
        }
        if (!maskFolder.exists) {
            alert("mask 文件夹不存在：\n" + maskFolder.fsName);
            return;
        }

        var outputFolder = new Folder(imageFolder.fsName + "/inpaint_overlay_psds");
        if (!outputFolder.exists) {
            outputFolder.create();
        }
        var pngFolder = null;
        if (exportPng) {
            pngFolder = new Folder(outputFolder.fsName + "/png");
            if (!pngFolder.exists) {
                pngFolder.create();
            }
        }

        var imageFiles = imageFolder.getFiles(function (file) {
            if (!(file instanceof File)) return false;
            return /\.(psd|png|jpg|jpeg|tif|tiff|bmp)$/i.test(file.name);
        });

        var processed = 0;
        var saved = 0;
        var pngSaved = 0;
        var withOther = [];
        var withoutOther = [];
        var skipped = [];
        var pngErrors = [];
        var actionExecuted = [];
        var actionErrors = [];

        for (var i = 0; i < imageFiles.length; i++) {
            var imageFile = imageFiles[i];
            var stem = stripExtension(imageFile.name);
            var maskFile = findMaskFile(maskFolder, stem);

            if (!maskFile) {
                skipped.push(imageFile.name + "：缺少同名 mask");
                continue;
            }

            processed++;
            var outDoc = null;
            try {
                outDoc = createOutputDocument(imageFile);
                var result = createInpaintOverlay(outDoc, maskFile);

                if (result.hasOtherRegion && runActionWhenOther) {
                    try {
                        app.activeDocument = outDoc;
                        app.doAction(settings.actionName, settings.actionSetName);
                        actionExecuted.push(imageFile.name);
                    } catch (actionErr) {
                        actionErrors.push(imageFile.name + "：" + actionErr.message);
                    }
                }

                var outFile = new File(outputFolder.fsName + "/" + stem + ".psd");
                saveDocument(outDoc, outFile);
                saved++;

                if (exportPng) {
                    try {
                        var pngFile = new File(pngFolder.fsName + "/" + stem + ".png");
                        exportCompositePng(outDoc, pngFile);
                        pngSaved++;
                    } catch (pngErr) {
                        pngErrors.push(imageFile.name + "：" + pngErr.message);
                    }
                }

                if (result.hasOtherRegion) {
                    withOther.push(imageFile.name + " | blocks=" + result.blocksProcessed);
                } else {
                    withoutOther.push(imageFile.name + " | blocks=" + result.blocksProcessed);
                }
            } catch (err) {
                skipped.push(imageFile.name + "：" + err.message);
            } finally {
                if (outDoc) {
                    try {
                        app.activeDocument = outDoc;
                        outDoc.close(SaveOptions.DONOTSAVECHANGES);
                    } catch (closeErr) {
                    }
                }
            }
        }

        var reportFile = new File(outputFolder.fsName + "/other_channel_report.txt");
        writeReport(reportFile, imageFolder, maskFolder, imageFiles.length, processed, saved, exportPng, pngSaved, runActionWhenOther, settings, actionExecuted, actionErrors, withOther, withoutOther, skipped, pngErrors);

        var message = "批量生成完成。\n输出目录：\n" + outputFolder.fsName +
            "\n\n图片总数：" + imageFiles.length +
            "\n成功保存：" + saved +
            "\n含 OTHER_CHANNEL：" + withOther.length +
            "\n报告：\n" + reportFile.fsName;
        if (exportPng) {
            message += "\nPNG 输出：" + pngSaved + "\nPNG 目录：\n" + pngFolder.fsName;
        }
        if (runActionWhenOther) {
            message += "\n执行动作：" + actionExecuted.length + "\n动作失败：" + actionErrors.length;
        }

        if (skipped.length > 0) {
            message += "\n\n跳过/失败：" + skipped.length + " 个\n" + skipped.slice(0, 20).join("\n");
            if (skipped.length > 20) message += "\n...";
        }
        if (pngErrors.length > 0) {
            message += "\n\nPNG 输出失败：" + pngErrors.length + " 个\n" + pngErrors.slice(0, 20).join("\n");
            if (pngErrors.length > 20) message += "\n...";
        }
        if (actionErrors.length > 0) {
            message += "\n\n动作执行失败：" + actionErrors.length + " 个\n" + actionErrors.slice(0, 20).join("\n");
            if (actionErrors.length > 20) message += "\n...";
        }

        alert(message);
    } catch (e) {
        alert("Batch mask inpaint overlay failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function showSettingsDialog() {
        var actionSets = getActionSets();
        var dialog = new Window("dialog", "批量生成纯色文字涂白 PSD");
        dialog.orientation = "column";
        dialog.alignChildren = ["fill", "top"];
        dialog.spacing = 10;
        dialog.margins = 16;

        var imageGroup = dialog.add("group");
        imageGroup.orientation = "row";
        imageGroup.alignChildren = ["fill", "center"];
        imageGroup.add("statictext", undefined, "原图文件夹：");
        var imagePathInput = imageGroup.add("edittext", undefined, "");
        imagePathInput.characters = 52;
        var imageBrowseButton = imageGroup.add("button", undefined, "选择");

        var maskGroup = dialog.add("group");
        maskGroup.orientation = "row";
        maskGroup.alignChildren = ["fill", "center"];
        maskGroup.add("statictext", undefined, "mask 文件夹：");
        var maskPathInput = maskGroup.add("edittext", undefined, "");
        maskPathInput.characters = 52;
        var maskBrowseButton = maskGroup.add("button", undefined, "选择");

        var pngGroup = dialog.add("group");
        pngGroup.orientation = "row";
        pngGroup.alignChildren = ["left", "center"];
        var pngCheckbox = pngGroup.add("checkbox", undefined, "同时输出合成 PNG 到 inpaint_overlay_psds/png");
        pngCheckbox.value = false;

        var actionEnableGroup = dialog.add("group");
        actionEnableGroup.orientation = "row";
        actionEnableGroup.alignChildren = ["left", "center"];
        var actionCheckbox = actionEnableGroup.add("checkbox", undefined, "有 OTHER_CHANNEL 时执行动作");
        actionCheckbox.value = false;

        var actionSetGroup = dialog.add("group");
        actionSetGroup.orientation = "row";
        actionSetGroup.alignChildren = ["left", "center"];
        actionSetGroup.add("statictext", undefined, "动作组：");
        var setDropdown = actionSetGroup.add("dropdownlist", undefined, []);
        setDropdown.minimumSize.width = 260;

        var actionGroup = dialog.add("group");
        actionGroup.orientation = "row";
        actionGroup.alignChildren = ["left", "center"];
        actionGroup.add("statictext", undefined, "动作：");
        var actionDropdown = actionGroup.add("dropdownlist", undefined, []);
        actionDropdown.minimumSize.width = 260;

        var buttonGroup = dialog.add("group");
        buttonGroup.orientation = "row";
        buttonGroup.alignment = "right";
        var okButton = buttonGroup.add("button", undefined, "OK", { name: "ok" });
        buttonGroup.add("button", undefined, "Cancel", { name: "cancel" });

        for (var i = 0; i < actionSets.length; i++) {
            setDropdown.add("item", actionSets[i].name);
        }
        if (actionSets.length > 0) {
            setDropdown.selection = 0;
        }
        refreshActionDropdown();
        refreshActionControls();

        imageBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择原图文件夹");
            if (selected) {
                imagePathInput.text = selected.fsName;
                maskPathInput.text = selected.fsName + "/mask";
            }
        };

        maskBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择 mask 文件夹");
            if (selected) {
                maskPathInput.text = selected.fsName;
            }
        };

        actionCheckbox.onClick = function () {
            refreshActionControls();
        };

        setDropdown.onChange = function () {
            refreshActionDropdown();
        };

        okButton.onClick = function () {
            if (!trimString(imagePathInput.text)) {
                alert("请选择原图文件夹。");
                return;
            }
            if (!trimString(maskPathInput.text)) {
                alert("请选择 mask 文件夹。");
                return;
            }
            if (actionCheckbox.value && (!setDropdown.selection || !actionDropdown.selection)) {
                alert("请先在 Photoshop Actions 面板载入动作，并选择动作组和动作。");
                return;
            }
            dialog.close(1);
        };

        if (dialog.show() !== 1) return null;

        var selectedSet = setDropdown.selection ? actionSets[setDropdown.selection.index] : null;
        var selectedAction = selectedSet && actionDropdown.selection ? selectedSet.actions[actionDropdown.selection.index] : null;

        return {
            imageFolder: trimString(imagePathInput.text),
            maskFolder: trimString(maskPathInput.text),
            exportPng: pngCheckbox.value,
            runActionWhenOther: actionCheckbox.value,
            actionSetName: selectedSet ? selectedSet.name : "",
            actionName: selectedAction ? selectedAction.name : ""
        };

        function refreshActionDropdown() {
            actionDropdown.removeAll();
            if (!setDropdown.selection) return;

            var selectedSet = actionSets[setDropdown.selection.index];
            for (var j = 0; j < selectedSet.actions.length; j++) {
                actionDropdown.add("item", selectedSet.actions[j].name);
            }
            if (selectedSet.actions.length > 0) {
                actionDropdown.selection = 0;
            }
        }

        function refreshActionControls() {
            var enabled = actionCheckbox.value && actionSets.length > 0;
            setDropdown.enabled = enabled;
            actionDropdown.enabled = enabled;
        }
    }

    function createOutputDocument(originalFile) {
        var srcDoc = app.open(originalFile);
        var docName = stripExtension(originalFile.name);

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

    function createInpaintOverlay(outDoc, maskFile) {
        var bgLayer = outDoc.artLayers.getByName("bg");
        var inpaintLayer = outDoc.artLayers.add();
        inpaintLayer.name = "bubble";
        inpaintLayer.move(bgLayer, ElementPlacement.PLACEBEFORE);

        var textChannel = pasteMaskIntoAlpha(outDoc, maskFile);
        var boxes = makeBoxesFromMaskSelection(outDoc, textChannel);
        boxes = mergeBoxes(boxes, GROUP_MERGE_PX, outDoc.width.as("px"), outDoc.height.as("px"));

        if (boxes.length === 0) {
            cleanupChannels(outDoc, [textChannel]);
            throw new Error("mask 中未找到文字区域");
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

        return {
            hasOtherRegion: hasOtherRegion,
            blocksProcessed: boxes.length
        };
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

        return dark < light;
    }

    function makeBoxesFromMaskSelection(doc, textChannel) {
        setRGBChannels(doc);
        doc.selection.load(textChannel, SelectionType.REPLACE);

        if (!hasSelection(doc)) return [];

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
        var isStrictSolid = maxSpread <= SOLID_P90_P10_MAX && minPeakRatio >= SOLID_PEAK_RATIO_MIN;
        var isWhiteDominant =
            r.peakValue >= WHITE_DOMINANT_MIN &&
            g.peakValue >= WHITE_DOMINANT_MIN &&
            b.peakValue >= WHITE_DOMINANT_MIN &&
            minPeakRatio >= WHITE_PEAK_RATIO_MIN;

        return {
            isSolid: isStrictSolid || isWhiteDominant,
            maxSpread: maxSpread,
            minPeakRatio: minPeakRatio,
            isWhiteDominant: isWhiteDominant
        };
    }

    function analyzeHistogram(hist) {
        var total = 0;
        var maxCount = 0;
        var peakValue = 0;
        var p10 = 0;
        var p90 = 255;

        for (var i = 0; i < 256; i++) {
            total += hist[i];
            if (hist[i] > maxCount) {
                maxCount = hist[i];
                peakValue = i;
            }
        }

        if (total <= 0) {
            return { spread: 255, peakRatio: 0, peakValue: 0 };
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
            peakRatio: maxCount / total,
            peakValue: peakValue
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

    function saveDocument(doc, outFile) {
        app.activeDocument = doc;
        setRGBChannels(doc);

        var saveOptions = new PhotoshopSaveOptions();
        saveOptions.alphaChannels = true;
        saveOptions.layers = true;
        doc.saveAs(outFile, saveOptions, true, Extension.LOWERCASE);
    }

    function exportCompositePng(doc, pngFile) {
        app.activeDocument = doc;
        setRGBChannels(doc);
        if (doc.mode !== DocumentMode.RGB) {
            doc.changeMode(ChangeMode.RGB);
        }
        doc.flatten();

        var pngOptions = new PNGSaveOptions();
        doc.saveAs(pngFile, pngOptions, true, Extension.LOWERCASE);
    }

    function getActionSets() {
        var sets = [];
        var index = 1;

        while (true) {
            var ref = new ActionReference();
            ref.putIndex(app.charIDToTypeID("ASet"), index);

            try {
                var desc = app.executeActionGet(ref);
                var name = desc.getString(app.charIDToTypeID("Nm  "));
                var count = 0;
                if (desc.hasKey(app.charIDToTypeID("NmbC"))) {
                    count = desc.getInteger(app.charIDToTypeID("NmbC"));
                }

                var actions = getActionsInSet(index, count);
                if (actions.length > 0) {
                    sets.push({
                        index: index,
                        name: name,
                        actions: actions
                    });
                }
                index++;
            } catch (e) {
                break;
            }
        }

        return sets;
    }

    function getActionsInSet(setIndex, count) {
        var actions = [];

        for (var i = 1; i <= count; i++) {
            var ref = new ActionReference();
            ref.putIndex(app.charIDToTypeID("Actn"), i);
            ref.putIndex(app.charIDToTypeID("ASet"), setIndex);

            try {
                var desc = app.executeActionGet(ref);
                actions.push({
                    index: i,
                    name: desc.getString(app.charIDToTypeID("Nm  "))
                });
            } catch (e) {
            }
        }

        return actions;
    }

    function writeReport(reportFile, imageFolder, maskFolder, totalImages, processed, saved, exportPng, pngSaved, runActionWhenOther, settings, actionExecuted, actionErrors, withOther, withoutOther, skipped, pngErrors) {
        reportFile.encoding = "UTF-8";
        if (!reportFile.open("w")) {
            alert("无法写入报告：\n" + reportFile.fsName);
            return;
        }

        reportFile.writeln("Batch mask inpaint overlay report");
        reportFile.writeln("Generated at: " + formatDate(new Date()));
        reportFile.writeln("Image folder: " + imageFolder.fsName);
        reportFile.writeln("Mask folder: " + maskFolder.fsName);
        reportFile.writeln("");
        reportFile.writeln("Total image files: " + totalImages);
        reportFile.writeln("Images with matching mask: " + processed);
        reportFile.writeln("Saved PSD files: " + saved);
        reportFile.writeln("Export PNG enabled: " + (exportPng ? "yes" : "no"));
        reportFile.writeln("Saved PNG files: " + pngSaved);
        reportFile.writeln("Run action when OTHER_CHANNEL exists: " + (runActionWhenOther ? "yes" : "no"));
        if (runActionWhenOther) {
            reportFile.writeln("Action set: " + settings.actionSetName);
            reportFile.writeln("Action: " + settings.actionName);
        }
        reportFile.writeln("Action executed: " + actionExecuted.length);
        reportFile.writeln("Action failed: " + actionErrors.length);
        reportFile.writeln("PSD files with OTHER_CHANNEL: " + withOther.length);
        reportFile.writeln("PSD files without OTHER_CHANNEL: " + withoutOther.length);
        reportFile.writeln("Skipped or failed: " + skipped.length);
        reportFile.writeln("PNG export failed: " + pngErrors.length);
        reportFile.writeln("");

        reportFile.writeln("[WITH_OTHER_CHANNEL]");
        writeLines(reportFile, withOther);
        reportFile.writeln("");

        reportFile.writeln("[WITHOUT_OTHER_CHANNEL]");
        writeLines(reportFile, withoutOther);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_EXECUTED]");
        writeLines(reportFile, actionExecuted);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_FAILED]");
        writeLines(reportFile, actionErrors);
        reportFile.writeln("");

        reportFile.writeln("[SKIPPED_OR_FAILED]");
        writeLines(reportFile, skipped);
        reportFile.writeln("");

        reportFile.writeln("[PNG_EXPORT_FAILED]");
        writeLines(reportFile, pngErrors);

        reportFile.close();
    }

    function writeLines(file, lines) {
        if (lines.length === 0) {
            file.writeln("(none)");
            return;
        }
        for (var i = 0; i < lines.length; i++) {
            file.writeln(lines[i]);
        }
    }

    function findMaskFile(maskFolder, stem) {
        var extensions = [".psd", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"];
        for (var i = 0; i < extensions.length; i++) {
            var file = new File(maskFolder.fsName + "/" + stem + extensions[i]);
            if (file.exists) return file;
        }
        return null;
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
        if (v && v.as) return v.as("px");
        return Number(v);
    }

    function stripExtension(name) {
        return name.replace(/\.[^\.]+$/, "");
    }

    function trimString(value) {
        return value.replace(/^\s+|\s+$/g, "");
    }

    function pad2(value) {
        return value < 10 ? "0" + value : String(value);
    }

    function formatDate(date) {
        return date.getFullYear() + "-" +
            pad2(date.getMonth() + 1) + "-" +
            pad2(date.getDate()) + " " +
            pad2(date.getHours()) + ":" +
            pad2(date.getMinutes()) + ":" +
            pad2(date.getSeconds());
    }
})();
