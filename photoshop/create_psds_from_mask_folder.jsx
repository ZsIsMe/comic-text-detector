/*
Create PSD files from an image folder and a selected mask folder.

Run in Photoshop:
File > Scripts > Browse... > create_psds_from_mask_folder.jsx

Dialog fields:
- Image folder
- Mask folder, auto-filled as <image folder>/mask after selecting image folder
- Alpha channel name, default "TEXT_MASK"

Output:
<image folder>/inpainted_psds/<image name>.psd
*/

#target photoshop

(function () {
    app.bringToFront();

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    var settings = showSettingsDialog();
    if (!settings) {
        app.preferences.rulerUnits = oldRulerUnits;
        return;
    }

    var imageFolder = new Folder(settings.imageFolder);
    var maskFolder = new Folder(settings.maskFolder);
    var channelName = settings.channelName;

    if (!imageFolder.exists) {
        alert("原图文件夹不存在：\n" + imageFolder.fsName);
        app.preferences.rulerUnits = oldRulerUnits;
        return;
    }

    if (!maskFolder.exists) {
        alert("mask 文件夹不存在：\n" + maskFolder.fsName);
        app.preferences.rulerUnits = oldRulerUnits;
        return;
    }

    var outputFolder = new Folder(imageFolder.fsName + "/inpainted_psds");
    if (!outputFolder.exists) {
        outputFolder.create();
    }

    var imageFiles = imageFolder.getFiles(function (file) {
        if (!(file instanceof File)) {
            return false;
        }
        return /\.(png|jpg|jpeg|tif|tiff|bmp)$/i.test(file.name);
    });

    var made = 0;
    var skipped = [];

    for (var i = 0; i < imageFiles.length; i++) {
        var imageFile = imageFiles[i];
        var stem = imageFile.name.replace(/\.[^.]+$/, "");
        var maskFile = findMaskFile(maskFolder, stem);

        if (!maskFile) {
            skipped.push(imageFile.name + "：缺少同名 mask");
            continue;
        }

        var doc = null;
        try {
            doc = app.open(imageFile);
            renameActiveLayer(doc, "original");

            importMaskAsAlpha(doc, maskFile, channelName);

            app.activeDocument = doc;
            doc.activeChannels = doc.componentChannels;

            var psdFile = new File(outputFolder.fsName + "/" + stem + ".psd");
            var saveOptions = new PhotoshopSaveOptions();
            saveOptions.alphaChannels = true;
            saveOptions.layers = true;
            doc.saveAs(psdFile, saveOptions, true, Extension.LOWERCASE);
            made++;
        } catch (err) {
            skipped.push(imageFile.name + "：" + err.message);
        } finally {
            if (doc) {
                app.activeDocument = doc;
                doc.close(SaveOptions.DONOTSAVECHANGES);
            }
        }
    }

    app.preferences.rulerUnits = oldRulerUnits;

    var message = "PSD 生成完成：" + made + " 个\n输出目录：\n" + outputFolder.fsName;
    if (skipped.length > 0) {
        message += "\n\n跳过/失败：" + skipped.length + " 个\n" + skipped.slice(0, 20).join("\n");
        if (skipped.length > 20) {
            message += "\n...";
        }
    }
    alert(message);
})();

function showSettingsDialog() {
    var dialog = new Window("dialog", "生成 PSD Alpha Channel");
    dialog.orientation = "column";
    dialog.alignChildren = ["fill", "top"];
    dialog.spacing = 10;
    dialog.margins = 16;

    var imageGroup = dialog.add("group");
    imageGroup.orientation = "row";
    imageGroup.alignChildren = ["fill", "center"];
    imageGroup.add("statictext", undefined, "原图文件夹：");
    var imagePathInput = imageGroup.add("edittext", undefined, "");
    imagePathInput.characters = 48;
    var imageBrowseButton = imageGroup.add("button", undefined, "选择");

    var maskGroup = dialog.add("group");
    maskGroup.orientation = "row";
    maskGroup.alignChildren = ["fill", "center"];
    maskGroup.add("statictext", undefined, "mask 文件夹：");
    var maskPathInput = maskGroup.add("edittext", undefined, "");
    maskPathInput.characters = 48;
    var maskBrowseButton = maskGroup.add("button", undefined, "选择");

    var channelGroup = dialog.add("group");
    channelGroup.orientation = "row";
    channelGroup.alignChildren = ["left", "center"];
    channelGroup.add("statictext", undefined, "mask 通道名称：");
    var channelInput = channelGroup.add("edittext", undefined, "TEXT_MASK");
    channelInput.characters = 24;

    var buttonGroup = dialog.add("group");
    buttonGroup.orientation = "row";
    buttonGroup.alignment = "right";
    var okButton = buttonGroup.add("button", undefined, "OK", { name: "ok" });
    buttonGroup.add("button", undefined, "Cancel", { name: "cancel" });

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

    okButton.onClick = function () {
        if (!trimString(imagePathInput.text)) {
            alert("请选择原图文件夹。");
            return;
        }
        if (!trimString(maskPathInput.text)) {
            alert("请选择 mask 文件夹。");
            return;
        }
        if (!trimString(channelInput.text)) {
            alert("请输入 mask 通道名称。");
            return;
        }
        dialog.close(1);
    };

    if (dialog.show() !== 1) {
        return null;
    }

    return {
        imageFolder: trimString(imagePathInput.text),
        maskFolder: trimString(maskPathInput.text),
        channelName: trimString(channelInput.text)
    };
}

function trimString(value) {
    return value.replace(/^\s+|\s+$/g, "");
}

function findMaskFile(maskFolder, stem) {
    var extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"];
    for (var i = 0; i < extensions.length; i++) {
        var file = new File(maskFolder.fsName + "/" + stem + extensions[i]);
        if (file.exists) {
            return file;
        }
    }
    return null;
}

function renameActiveLayer(doc, layerName) {
    try {
        doc.activeLayer.name = layerName;
    } catch (err) {
    }
}

function importMaskAsAlpha(targetDoc, maskFile, channelName) {
    app.activeDocument = targetDoc;
    removeAlphaChannelIfExists(targetDoc, channelName);

    var maskDoc = app.open(maskFile);
    var targetWidth = Math.round(targetDoc.width.as("px"));
    var targetHeight = Math.round(targetDoc.height.as("px"));
    var maskWidth = Math.round(maskDoc.width.as("px"));
    var maskHeight = Math.round(maskDoc.height.as("px"));

    if (targetWidth !== maskWidth || targetHeight !== maskHeight) {
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
    targetDoc.paste();
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
