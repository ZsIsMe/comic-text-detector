/*
Create PSD files from solid_inpaint outputs.

Run in Photoshop:
File > Scripts > Browse... > create_psds_from_outputs.jsx

Expected input layout:
<image folder>/ctd_inpainted/mask/<name>.png
<image folder>/ctd_inpainted/other_mask/<name>.png
<image folder>/ctd_inpainted/inpainted/<name>.png

Output:
<image folder>/ctd_inpainted/psd/<name>.psd

Each PSD contains:
- bg
- overlay-manual
- TEXT_CHANNEL
- OTHER_CHANNEL
*/

#target photoshop

(function () {
    app.bringToFront();

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        var settings = showSettingsDialog();
        if (!settings) return;

        var imageFolder = new Folder(settings.imageFolder);
        var outputRoot = new Folder(settings.outputRoot);
        var maskFolder = new Folder(outputRoot.fsName + "/mask");
        var otherMaskFolder = new Folder(outputRoot.fsName + "/other_mask");
        var overlayFolder = new Folder(outputRoot.fsName + "/inpainted");
        var psdFolder = new Folder(outputRoot.fsName + "/psd");

        if (!imageFolder.exists) {
            alert("原图文件夹不存在：\n" + imageFolder.fsName);
            return;
        }
        if (!maskFolder.exists) {
            alert("mask 文件夹不存在：\n" + maskFolder.fsName);
            return;
        }
        if (!otherMaskFolder.exists) {
            alert("other_mask 文件夹不存在：\n" + otherMaskFolder.fsName);
            return;
        }
        if (!overlayFolder.exists) {
            alert("inpainted 文件夹不存在：\n" + overlayFolder.fsName);
            return;
        }
        if (!psdFolder.exists) {
            psdFolder.create();
        }

        var imageFiles = imageFolder.getFiles(function (file) {
            if (!(file instanceof File)) return false;
            return /\.(png|jpg|jpeg|tif|tiff|bmp|psd)$/i.test(file.name);
        });
        imageFiles.sort(function (a, b) {
            var an = a.name.toLowerCase();
            var bn = b.name.toLowerCase();
            if (an < bn) return -1;
            if (an > bn) return 1;
            return 0;
        });

        var made = 0;
        var actionRun = [];
        var actionErrors = [];
        var skipped = [];

        for (var i = 0; i < imageFiles.length; i++) {
            var imageFile = imageFiles[i];
            var stem = stripExtension(imageFile.name);
            var maskFile = findMaskFile(maskFolder, stem);
            var otherMaskFile = findMaskFile(otherMaskFolder, stem);
            var overlayFile = findMaskFile(overlayFolder, stem);

            if (!maskFile) {
                skipped.push(imageFile.name + "：缺少 mask");
                continue;
            }
            if (!otherMaskFile) {
                skipped.push(imageFile.name + "：缺少 other_mask");
                continue;
            }
            if (!overlayFile) {
                skipped.push(imageFile.name + "：缺少 inpainted overlay");
                continue;
            }

            var doc = null;
            try {
                doc = createDocument(imageFile);
                importOverlayLayer(doc, overlayFile, "overlay-manual");
                importMaskAsAlpha(doc, maskFile, "TEXT_CHANNEL");
                importMaskAsAlpha(doc, otherMaskFile, "OTHER_CHANNEL");

                if (settings.runAction && hasChannel(doc, "OTHER_CHANNEL")) {
                    try {
                        app.activeDocument = doc;
                        app.doAction(settings.actionName, settings.actionSetName);
                        actionRun.push(imageFile.name);
                    } catch (actionErr) {
                        actionErrors.push(imageFile.name + "：" + actionErr.message);
                    }
                }

                app.activeDocument = doc;
                setRGBChannels(doc);
                var psdFile = new File(psdFolder.fsName + "/" + stem + ".psd");
                var saveOptions = new PhotoshopSaveOptions();
                saveOptions.alphaChannels = true;
                saveOptions.layers = true;
                doc.saveAs(psdFile, saveOptions, true, Extension.LOWERCASE);
                made++;
            } catch (err) {
                skipped.push(imageFile.name + "：" + err.message);
            } finally {
                if (doc) {
                    try {
                        app.activeDocument = doc;
                        doc.close(SaveOptions.DONOTSAVECHANGES);
                    } catch (closeErr) {
                    }
                }
            }
        }

        writeReport(
            new File(psdFolder.fsName + "/create_psds_report.txt"),
            imageFolder,
            outputRoot,
            imageFiles.length,
            made,
            settings,
            actionRun,
            actionErrors,
            skipped
        );

        var message = "PSD 生成完成：" + made + " 个\n输出目录：\n" + psdFolder.fsName;
        if (settings.runAction) {
            message += "\n执行动作：" + actionRun.length + "\n动作失败：" + actionErrors.length;
        }
        if (skipped.length > 0) {
            message += "\n\n跳过/失败：" + skipped.length + " 个\n" + skipped.slice(0, 20).join("\n");
            if (skipped.length > 20) message += "\n...";
        }
        if (actionErrors.length > 0) {
            message += "\n\n动作失败前 20 个：\n" + actionErrors.slice(0, 20).join("\n");
            if (actionErrors.length > 20) message += "\n...";
        }
        alert(message);
    } catch (e) {
        alert("Create solid inpaint PSDs failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function showSettingsDialog() {
        var actionSets = getActionSets();
        var dialog = new Window("dialog", "生成 solid_inpaint PSD");
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

        var outputGroup = dialog.add("group");
        outputGroup.orientation = "row";
        outputGroup.alignChildren = ["fill", "center"];
        outputGroup.add("statictext", undefined, "ctd_inpainted：");
        var outputPathInput = outputGroup.add("edittext", undefined, "");
        outputPathInput.characters = 52;
        var outputBrowseButton = outputGroup.add("button", undefined, "选择");

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
                outputPathInput.text = selected.fsName + "/ctd_inpainted";
            }
        };

        outputBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择 ctd_inpainted 文件夹");
            if (selected) {
                outputPathInput.text = selected.fsName;
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
            if (!trimString(outputPathInput.text)) {
                alert("请选择 ctd_inpainted 文件夹。");
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
            outputRoot: trimString(outputPathInput.text),
            runAction: actionCheckbox.value,
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

    function createDocument(imageFile) {
        var srcDoc = app.open(imageFile);
        var docName = stripExtension(imageFile.name);
        var doc = srcDoc.duplicate(docName, true);
        srcDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = doc;
        if (doc.mode !== DocumentMode.RGB) {
            doc.changeMode(ChangeMode.RGB);
        }
        doc.activeLayer = doc.layers[0];
        doc.activeLayer.name = "bg";
        return doc;
    }

    function importOverlayLayer(targetDoc, overlayFile, layerName) {
        var overlayDoc = app.open(overlayFile);
        var targetWidth = Math.round(targetDoc.width.as("px"));
        var targetHeight = Math.round(targetDoc.height.as("px"));
        if (Math.round(overlayDoc.width.as("px")) !== targetWidth ||
            Math.round(overlayDoc.height.as("px")) !== targetHeight) {
            overlayDoc.close(SaveOptions.DONOTSAVECHANGES);
            throw new Error(layerName + " 尺寸不一致");
        }

        overlayDoc.selection.selectAll();
        overlayDoc.selection.copy();
        overlayDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = targetDoc;
        setRGBChannels(targetDoc);
        targetDoc.paste();
        targetDoc.activeLayer.name = layerName;

        try {
            var bgLayer = targetDoc.artLayers.getByName("bg");
            targetDoc.activeLayer.move(bgLayer, ElementPlacement.PLACEBEFORE);
        } catch (e) {
        }
    }

    function importMaskAsAlpha(targetDoc, maskFile, channelName) {
        app.activeDocument = targetDoc;
        removeAlphaChannelIfExists(targetDoc, channelName);

        var maskDoc = app.open(maskFile);
        var targetWidth = Math.round(targetDoc.width.as("px"));
        var targetHeight = Math.round(targetDoc.height.as("px"));
        if (Math.round(maskDoc.width.as("px")) !== targetWidth ||
            Math.round(maskDoc.height.as("px")) !== targetHeight) {
            maskDoc.close(SaveOptions.DONOTSAVECHANGES);
            throw new Error(channelName + " 尺寸不一致");
        }

        maskDoc.selection.selectAll();
        maskDoc.selection.copy();
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = targetDoc;
        var alpha = targetDoc.channels.add();
        alpha.name = channelName;
        targetDoc.activeChannels = [alpha];
        targetDoc.selection.selectAll();
        targetDoc.paste();
        targetDoc.selection.deselect();
        setRGBChannels(targetDoc);
    }

    function hasChannel(doc, channelName) {
        for (var i = 0; i < doc.channels.length; i++) {
            if (doc.channels[i].name === channelName) return true;
        }
        return false;
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

    function writeReport(reportFile, imageFolder, outputRoot, total, made, settings, actionRun, actionErrors, skipped) {
        reportFile.encoding = "UTF-8";
        if (!reportFile.open("w")) {
            alert("无法写入报告：\n" + reportFile.fsName);
            return;
        }
        reportFile.writeln("Create solid_inpaint PSD report");
        reportFile.writeln("Generated at: " + formatDate(new Date()));
        reportFile.writeln("Image folder: " + imageFolder.fsName);
        reportFile.writeln("ctd_inpainted folder: " + outputRoot.fsName);
        reportFile.writeln("Total image files: " + total);
        reportFile.writeln("Saved PSD files: " + made);
        reportFile.writeln("Run action: " + (settings.runAction ? "yes" : "no"));
        if (settings.runAction) {
            reportFile.writeln("Action set: " + settings.actionSetName);
            reportFile.writeln("Action: " + settings.actionName);
        }
        reportFile.writeln("Action executed: " + actionRun.length);
        reportFile.writeln("Action failed: " + actionErrors.length);
        reportFile.writeln("Skipped or failed: " + skipped.length);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_EXECUTED]");
        writeLines(reportFile, actionRun);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_FAILED]");
        writeLines(reportFile, actionErrors);
        reportFile.writeln("");

        reportFile.writeln("[SKIPPED_OR_FAILED]");
        writeLines(reportFile, skipped);
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

    function findMaskFile(folder, stem) {
        var extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".psd"];
        for (var i = 0; i < extensions.length; i++) {
            var file = new File(folder.fsName + "/" + stem + extensions[i]);
            if (file.exists) return file;
        }
        return null;
    }

    function removeAlphaChannelIfExists(doc, channelName) {
        for (var i = doc.channels.length - 1; i >= 0; i--) {
            var channel = doc.channels[i];
            if (channel.name === channelName) {
                channel.remove();
                return;
            }
        }
    }

    function setRGBChannels(doc) {
        doc.activeChannels = [doc.channels[0], doc.channels[1], doc.channels[2]];
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
