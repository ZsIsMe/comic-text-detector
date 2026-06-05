/*
Create PSD files with mask PNGs imported as alpha channels.

Run in Photoshop:
File > Scripts > Browse... > create_psd_with_mask_channels.jsx

Expected folder layout:
<image folder>/
  page.png / page.jpg / ...
  ctd/progressing/mask/page.png
  ctd/progressing/block_mask/page.png
  ctd/progressing/other_mask/page.png

Output:
<image folder>/psd/page.psd
*/

#target photoshop

(function () {
    app.bringToFront();

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    var imageFolder = Folder.selectDialog("选择图片文件夹");
    if (!imageFolder) {
        app.preferences.rulerUnits = oldRulerUnits;
        return;
    }

    var psdFolder = new Folder(imageFolder.fsName + "/psd");
    if (!psdFolder.exists) {
        psdFolder.create();
    }

    var maskFolder = new Folder(imageFolder.fsName + "/ctd/progressing/mask");
    var blockMaskFolder = new Folder(imageFolder.fsName + "/ctd/progressing/block_mask");
    var otherMaskFolder = new Folder(imageFolder.fsName + "/ctd/progressing/other_mask");

    if (!maskFolder.exists || !blockMaskFolder.exists || !otherMaskFolder.exists) {
        alert("找不到 mask 文件夹。需要：\n" +
            maskFolder.fsName + "\n" +
            blockMaskFolder.fsName + "\n" +
            otherMaskFolder.fsName);
        app.preferences.rulerUnits = oldRulerUnits;
        return;
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
        var maskFile = new File(maskFolder.fsName + "/" + stem + ".png");
        var blockMaskFile = new File(blockMaskFolder.fsName + "/" + stem + ".png");
        var otherMaskFile = new File(otherMaskFolder.fsName + "/" + stem + ".png");

        if (!maskFile.exists || !blockMaskFile.exists || !otherMaskFile.exists) {
            skipped.push(imageFile.name + "：缺少同名 mask");
            continue;
        }

        var doc = app.open(imageFile);
        doc.activeLayer.name = "original";

        try {
            importMaskAsAlpha(doc, maskFile, "TEXT_MASK");
            importMaskAsAlpha(doc, blockMaskFile, "BLOCK_TEXT_MASK");
            importMaskAsAlpha(doc, otherMaskFile, "OTHER_TEXT_MASK");

            app.activeDocument = doc;
            doc.activeChannels = doc.componentChannels;

            var psdFile = new File(psdFolder.fsName + "/" + stem + ".psd");
            var saveOptions = new PhotoshopSaveOptions();
            saveOptions.alphaChannels = true;
            saveOptions.layers = true;
            doc.saveAs(psdFile, saveOptions, true, Extension.LOWERCASE);
            made++;
        } catch (err) {
            skipped.push(imageFile.name + "：" + err.message);
        } finally {
            app.activeDocument = doc;
            doc.close(SaveOptions.DONOTSAVECHANGES);
        }
    }

    app.preferences.rulerUnits = oldRulerUnits;

    var message = "PSD 生成完成：" + made + " 个";
    if (skipped.length > 0) {
        message += "\n\n跳过/失败：" + skipped.length + " 个\n" + skipped.slice(0, 20).join("\n");
        if (skipped.length > 20) {
            message += "\n...";
        }
    }
    alert(message);
})();

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
