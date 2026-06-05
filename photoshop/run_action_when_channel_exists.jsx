/*
Batch run a Photoshop Action only when a PSD contains a named channel.

Run in Photoshop:
File > Scripts > Browse... > run_action_when_channel_exists.jsx

Input:
- PSD folder
- Channel name, default "OTHER_CHANNEL"
- Photoshop Action Set and Action, loaded from the current Actions panel

Behavior:
- If the PSD contains the channel, run the selected action, then close.
- If the PSD does not contain the channel, close without saving.
- Per-file errors are written to a report and do not stop the batch.

Output:
<PSD folder>/channel_action_report.txt
*/

#target photoshop

(function () {
    app.bringToFront();

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        var actionSets = getActionSets();
        if (actionSets.length === 0) {
            alert("没有找到 Photoshop 动作组。\n请先在 Actions 面板中载入至少一个动作组。");
            return;
        }

        var settings = showSettingsDialog(actionSets);
        if (!settings) return;

        var psdFolder = new Folder(settings.psdFolder);
        if (!psdFolder.exists) {
            alert("PSD 文件夹不存在：\n" + psdFolder.fsName);
            return;
        }

        var psdFiles = psdFolder.getFiles(function (file) {
            if (!(file instanceof File)) return false;
            return /\.psd$/i.test(file.name);
        });
        psdFiles.sort(function (a, b) {
            var an = a.name.toLowerCase();
            var bn = b.name.toLowerCase();
            if (an < bn) return -1;
            if (an > bn) return 1;
            return 0;
        });

        var actionRun = [];
        var noChannel = [];
        var failed = [];

        for (var i = 0; i < psdFiles.length; i++) {
            processOnePsd(psdFiles[i], settings, actionRun, noChannel, failed);
        }

        var reportFile = new File(psdFolder.fsName + "/channel_action_report.txt");
        writeReport(reportFile, psdFolder, settings, psdFiles.length, actionRun, noChannel, failed);

        var message = "整理完成。\nPSD 总数：" + psdFiles.length +
            "\n执行动作：" + actionRun.length +
            "\n没有目标 Channel：" + noChannel.length +
            "\n失败：" + failed.length +
            "\n报告：\n" + reportFile.fsName;

        if (failed.length > 0) {
            message += "\n\n失败前 20 个：\n" + failed.slice(0, 20).join("\n");
            if (failed.length > 20) message += "\n...";
        }

        alert(message);
    } catch (e) {
        alert("Batch channel action failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function showSettingsDialog(actionSets) {
        var dialog = new Window("dialog", "按 Channel 批量执行动作");
        dialog.orientation = "column";
        dialog.alignChildren = ["fill", "top"];
        dialog.spacing = 10;
        dialog.margins = 16;

        var folderGroup = dialog.add("group");
        folderGroup.orientation = "row";
        folderGroup.alignChildren = ["fill", "center"];
        folderGroup.add("statictext", undefined, "PSD 文件夹：");
        var folderInput = folderGroup.add("edittext", undefined, "");
        folderInput.characters = 52;
        var folderButton = folderGroup.add("button", undefined, "选择");

        var channelGroup = dialog.add("group");
        channelGroup.orientation = "row";
        channelGroup.alignChildren = ["left", "center"];
        channelGroup.add("statictext", undefined, "Channel 名称：");
        var channelInput = channelGroup.add("edittext", undefined, "OTHER_CHANNEL");
        channelInput.characters = 28;

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

        var saveGroup = dialog.add("group");
        saveGroup.orientation = "row";
        saveGroup.alignChildren = ["left", "center"];
        var saveCheckbox = saveGroup.add("checkbox", undefined, "执行动作后保存 PSD");
        saveCheckbox.value = true;

        var buttonGroup = dialog.add("group");
        buttonGroup.orientation = "row";
        buttonGroup.alignment = "right";
        var okButton = buttonGroup.add("button", undefined, "OK", { name: "ok" });
        buttonGroup.add("button", undefined, "Cancel", { name: "cancel" });

        for (var i = 0; i < actionSets.length; i++) {
            setDropdown.add("item", actionSets[i].name);
        }
        setDropdown.selection = 0;
        refreshActionDropdown();

        folderButton.onClick = function () {
            var selected = Folder.selectDialog("选择 PSD 文件夹");
            if (selected) {
                folderInput.text = selected.fsName;
            }
        };

        setDropdown.onChange = function () {
            refreshActionDropdown();
        };

        okButton.onClick = function () {
            if (!trimString(folderInput.text)) {
                alert("请选择 PSD 文件夹。");
                return;
            }
            if (!trimString(channelInput.text)) {
                alert("请输入 Channel 名称。");
                return;
            }
            if (!setDropdown.selection || !actionDropdown.selection) {
                alert("请选择动作组和动作。");
                return;
            }
            dialog.close(1);
        };

        if (dialog.show() !== 1) return null;

        var selectedSet = actionSets[setDropdown.selection.index];
        var selectedAction = selectedSet.actions[actionDropdown.selection.index];

        return {
            psdFolder: trimString(folderInput.text),
            channelName: trimString(channelInput.text),
            actionSetName: selectedSet.name,
            actionName: selectedAction.name,
            saveAfterAction: saveCheckbox.value
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
    }

    function processOnePsd(psdFile, settings, actionRun, noChannel, failed) {
        var doc = null;
        var actionStarted = false;

        try {
            doc = app.open(psdFile);
            if (!hasChannel(doc, settings.channelName)) {
                noChannel.push(psdFile.name);
                closeDoc(doc, SaveOptions.DONOTSAVECHANGES);
                doc = null;
                return;
            }

            actionStarted = true;
            app.activeDocument = doc;
            app.doAction(settings.actionName, settings.actionSetName);

            if (settings.saveAfterAction) {
                closeDoc(doc, SaveOptions.SAVECHANGES);
            } else {
                closeDoc(doc, SaveOptions.DONOTSAVECHANGES);
            }
            doc = null;
            actionRun.push(psdFile.name);
        } catch (err) {
            failed.push(psdFile.name + "：" + err.message + (actionStarted ? "（执行动作阶段）" : ""));
            if (doc) {
                try {
                    closeDoc(doc, SaveOptions.DONOTSAVECHANGES);
                } catch (closeErr) {
                    failed.push(psdFile.name + "：关闭失败：" + closeErr.message);
                }
            }
        }
    }

    function hasChannel(doc, channelName) {
        for (var i = 0; i < doc.channels.length; i++) {
            if (doc.channels[i].name === channelName) {
                return true;
            }
        }
        return false;
    }

    function closeDoc(doc, saveOption) {
        app.activeDocument = doc;
        doc.close(saveOption);
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

    function writeReport(reportFile, psdFolder, settings, total, actionRun, noChannel, failed) {
        reportFile.encoding = "UTF-8";
        if (!reportFile.open("w")) {
            alert("无法写入报告：\n" + reportFile.fsName);
            return;
        }

        reportFile.writeln("Batch channel action report");
        reportFile.writeln("Generated at: " + formatDate(new Date()));
        reportFile.writeln("PSD folder: " + psdFolder.fsName);
        reportFile.writeln("Channel name: " + settings.channelName);
        reportFile.writeln("Action set: " + settings.actionSetName);
        reportFile.writeln("Action: " + settings.actionName);
        reportFile.writeln("Save after action: " + (settings.saveAfterAction ? "yes" : "no"));
        reportFile.writeln("");
        reportFile.writeln("Total PSD files: " + total);
        reportFile.writeln("Action executed: " + actionRun.length);
        reportFile.writeln("Without channel: " + noChannel.length);
        reportFile.writeln("Failed: " + failed.length);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_EXECUTED]");
        writeLines(reportFile, actionRun);
        reportFile.writeln("");

        reportFile.writeln("[WITHOUT_CHANNEL]");
        writeLines(reportFile, noChannel);
        reportFile.writeln("");

        reportFile.writeln("[FAILED]");
        writeLines(reportFile, failed);

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
