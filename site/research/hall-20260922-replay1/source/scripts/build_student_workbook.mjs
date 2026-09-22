import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT = path.join(ROOT, 'outputs/01a0c1ce-44a4-79d3-ac49-a2bb3be060bb');
const QA = path.join(OUT, 'workbook-qa');
const DATA = path.join(ROOT, 'outputs/teaching/hall-20260922-replay1/data');
const require = createRequire(path.join(OUT, 'artifact-loader.cjs'));
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
await fs.mkdir(QA, { recursive: true });
const summary = JSON.parse(await fs.readFile(path.join(DATA, 'summary.json'), 'utf8'));
const dictionary = JSON.parse(await fs.readFile(path.join(DATA, 'feature_dictionary.json'), 'utf8'));

async function csv(name, stringColumns) {
  const source = await Workbook.fromCSV((await fs.readFile(path.join(DATA, name), 'utf8')).replace(/^\uFEFF/, ''), { sheetName: 'Import' });
  const grid = source.worksheets.getItemAt(0).getUsedRange().values;
  return [grid[0], ...grid.slice(1).map(row => row.map((v, i) => {
    if (stringColumns.includes(grid[0][i])) return v == null ? '' : String(v);
    if (v == null || v === '') return null;
    if (v === 'true') return true;
    if (v === 'false') return false;
    const n = Number(v);
    assert(Number.isFinite(n), `${name}: invalid number ${v}`);
    return n;
  }))];
}
const [episodes, samples, fits] = await Promise.all([
  csv('episodes.csv', ['phase', 'failure_reason']),
  csv('training_samples.csv', []),
  csv('training_history.csv', []),
]);
assert.equal(episodes.length - 1, 64);
assert.equal(samples.length - 1, 2400);
assert.equal(samples[0].length, 83);
assert.equal(fits.length - 1, 2);
assert.equal(samples.slice(1).filter(r => r[2] === true).length, 1600);

const wb = Workbook.create();
const overview = wb.worksheets.add('實驗摘要');
const episodesSheet = wb.worksheets.add('逐次評估');
const samplesSheet = wb.worksheets.add('訓練樣本');
const definitions = wb.worksheets.add('欄位與方法');
const INK = '#263C32', LIGHT = '#EDF2EE', GRAY = '#617168';
const FONT = 'Arial';
function range(sheet, row, col, rows, cols) { return sheet.getRangeByIndexes(row - 1, col - 1, rows, cols); }
function column(n) { let s=''; for (;n;n=Math.floor((n-1)/26)) s=String.fromCharCode(65+(n-1)%26)+s; return s; }
function base(sheet, rows, cols) {
  sheet.showGridLines = false;
  const used = range(sheet, 1, 1, rows, cols);
  used.format.font = { name: FONT, size: 11, color: '#253129' };
  used.format.rowHeight = 22;
  used.format.columnWidth = 16;
  used.format.verticalAlignment = 'center';
}
function title(sheet, text) {
  sheet.getRange('A2').values = [[text]];
  sheet.getRange('A2').format.font = { name: FONT, size: 16, bold: true, color: INK };
  sheet.getRange('A2:J2').format.rowHeight = 30;
  sheet.getRange('A3:J3').format.borders = { bottom: { style: 'thin', color: '#BDC9BF' } };
}
function header(sheet, address) {
  sheet.getRange(address).format = {
    fill: INK,
    font: { name: FONT, size: 11, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true,
    borders: { insideVertical: { style: 'thin', color: '#FFFFFF' } },
  };
}
function flatTable(sheet, grid, tableName, sourceNote, widths) {
  const rows = grid.length + 3, cols = grid[0].length;
  base(sheet, rows, cols);
  sheet.getRange('A1').values = [[sourceNote]];
  sheet.getRange('A1').format.font = { name: FONT, size: 11, color: GRAY, italic: true };
  sheet.getRange('A2').values = [['數字及 TRUE/FALSE 保留原始類型。第 4 列為欄名，使用篩選選擇資料。']];
  range(sheet, 4, 1, grid.length, cols).values = grid;
  const tableAddress = `A4:${column(cols)}${rows}`;
  const table = sheet.tables.add(tableAddress, true, tableName);
  table.showFilterButton = true;
  table.style = 'TableStyleLight8';
  header(sheet, `A4:${column(cols)}4`);
  range(sheet, 4, 1, 1, cols).format.rowHeight = 46;
  range(sheet, 5, 1, grid.length - 1, cols).format.horizontalAlignment = 'right';
  range(sheet, 5, 1, grid.length - 1, cols).setNumberFormat('0.000000');
  for (let i = 0; i < cols; i++) range(sheet, 1, i + 1, rows, 1).format.columnWidth = widths[i] ?? 17;
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(3);
  return rows;
}

flatTable(episodesSheet, episodes, 'EpisodeRecords', '來源：episodes.csv；run hall-20260922-replay1；64 次評估，包含重用驗證 seeds 的不同模型。', [23,14,14,12,12,12,15,21,20,23,23,19,16,17,17]);
episodesSheet.getRange('A5:A68').format.horizontalAlignment = 'left';
episodesSheet.getRange('H5:H68').format.horizontalAlignment = 'left';
episodesSheet.getRange('B5:C68').setNumberFormat('0');
episodesSheet.getRange('F5:F68').setNumberFormat('0');
episodesSheet.getRange('D5:E68').setNumberFormat('General');
episodesSheet.getRange('E5:E68').conditionalFormats.add('cellIs', {operator:'equal',formula:'TRUE',format:{fill:'#FCE2DF',font:{color:'#9C302A',bold:true}}});
episodesSheet.getRange('K5:K68').conditionalFormats.add('cellIs', {operator:'lessThan',formula:0,format:{fill:'#FCE2DF',font:{color:'#9C302A',bold:true}}});

flatTable(samplesSheet, samples, 'TrainingRecords', '來源：training_samples.csv / hall-demonstrations.npz；2,400 筆特徵及教師標籤。完整 RGB、逐筆 seed / 時間未保存。', [15,18,26]);
samplesSheet.getRange('A5:B2404').setNumberFormat('0');
samplesSheet.getRange('C5:C2404').setNumberFormat('General');

base(overview, 42, 11);
overview.tabColor = INK;
overview.getRange('A1:A42').format.columnWidth = 23;
overview.getRange('B1:B42').format.columnWidth = 33;
overview.getRange('C1:G42').format.columnWidth = 13;
overview.getRange('H1:J42').format.columnWidth = 20;
title(overview, '工廠避障訓練數據');
overview.getRange('A4').values = [['hall-20260922-replay1   2026-09-22   TUM Flight Hall']];
overview.getRange('A4').format.font = {name:FONT,size:11,color:GRAY};
overview.getRange('A6:J6').values = [['phase','評估條件','次數','到達次數','碰撞次數','到達率','碰撞率','平均時長 (s)','最小折線淨距 (m)','平均終點距離 (m)']];
header(overview,'A6:J6');
overview.getRange('A6:J6').format.rowHeight = 44;
for (let i = 0; i < summary.phases.length; i++) {
  const p = summary.phases[i], r = 7 + i;
  overview.getRange(`A${r}:B${r}`).values = [[p.phase,p.label]];
  overview.getRange(`C${r}:J${r}`).formulas = [[
    `=COUNTIFS('逐次評估'!$A$5:$A$68,$A${r})`,
    `=SUMPRODUCT(('逐次評估'!$A$5:$A$68=$A${r})*('逐次評估'!$D$5:$D$68=TRUE))`,
    `=SUMPRODUCT(('逐次評估'!$A$5:$A$68=$A${r})*('逐次評估'!$E$5:$E$68=TRUE))`,
    `=D${r}/C${r}`,`=E${r}/C${r}`,
    `=AVERAGEIFS('逐次評估'!$G$5:$G$68,'逐次評估'!$A$5:$A$68,$A${r})`,
    `=_xlfn.MINIFS('逐次評估'!$K$5:$K$68,'逐次評估'!$A$5:$A$68,$A${r})`,
    `=AVERAGEIFS('逐次評估'!$I$5:$I$68,'逐次評估'!$A$5:$A$68,$A${r})`,
  ]];
}
overview.getRange('B7:B11').format.wrapText = true;
overview.getRange('A7:J11').format.rowHeight = 38;
overview.getRange('C7:E11').setNumberFormat('0');
overview.getRange('F7:G11').setNumberFormat('0.0%');
overview.getRange('H7:J11').setNumberFormat('0.000');
overview.getRange('E7:E11').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#FCE2DF',font:{color:'#9C302A',bold:true}}});
overview.getRange('A13').values = [['平均時長包含失敗次數；64 次不可合併成一個獨立成功率。對照 20 次全數逾時，沒有到達。']];
overview.getRange('A14').values = [['原模型與第 0 輪在相同驗證組皆為 8/8，不能宣稱到達率提升。正式測試不參與選模。']];
overview.getRange('A16:D18').values = [['擬合輪次','累積擬合樣本','最後 epoch MSE','獲選'],...fits.slice(1)];
header(overview,'A16:D16');
overview.getRange('C17:C18').setNumberFormat('0.0000000');
overview.getRange('F16').values = [['來源：training_history.csv']];
overview.getRange('F17').values = [['只保存兩個訓練結束值。']];
overview.getRange('F18').values = [['沒有完整 60-epoch loss 曲線。']];
overview.getRange('A21:B21').values = [['資料量','筆數']];
header(overview,'A21:B21');
overview.getRange('A22:A24').values = [['全部新收集樣本'],['獲選模型擬合樣本'],['輸入特徵維數']];
overview.getRange('B22').formulas = [["=COUNT('訓練樣本'!$A$5:$A$2404)"]];
overview.getRange('B23').formulas = [['=IF(D17,B17,0)+IF(D18,B18,0)']];
overview.getRange('B24').values = [[summary.counts.observation_dimensions]];
overview.getRange('B22:B24').setNumberFormat('#,##0');
overview.getRange('D22').values = [['先收集 1,600 筆示範，再收集 800 筆 DAgger；第 1 輪驗證退步，未選用。']];
overview.getRange('A27').values = [['解讀範圍']];
overview.getRange('A27').format.font = {name:FONT,size:11,bold:true,color:INK};
const limits = [
 '一個靜態工廠局部場景，固定高度及偏航，單一手動圓柱碰撞代理。',
 '目標／速度輔助的水平導航；Flyvis 凍結，只微調決策網絡，沒有訓練整個果蠅大腦。',
 '20 個正式測試 seeds 並非 20 個新場景。最小折線淨距只描述記錄位置之間的直線。',
 '尚未驗證真實測距、全工廠碰撞、4DGS、校園、戶外、降落或真機。',
 '第三身畫面是已記錄測試的回放。詳細欄位、單位與缺失資料見「欄位與方法」。',
];
overview.getRange('A28:A32').values = limits.map(x=>[x]);

const protocol = [
 ['資料來源','summary.json、feature_dictionary.json、episodes.csv、training_samples.csv；本輪記錄於 2026-09-22。'],
 ['run_id',summary.run_id],
 ['到達判定','12 秒內進入目標 0.25 m 範圍。seconds 是每次模擬時長，含失敗。'],
 ['訓練方法','示範學習與純學習策略 DAgger；Flyvis 凍結。輸出為教師水平速度標籤。'],
 ['資料分組','validation-source / validation-0 / validation-1 各 8 次；audit / stale-vision 各 20 次。'],
 ['固定首幀對照','僅固定 72 個視覺特徵，其餘目標、速度、上一指令仍更新；audit 與對照使用相同 seeds。'],
 ['代理與包絡','圓柱半徑 0.45 m、高 1.55 m；水平機身包絡半徑 0.09 m，是保守包絡而非實際機身寬度。'],
 ['淨距單位','min_sampled_clearance_m 是取樣位置淨距；min_polyline_clearance_m 是相鄰記錄點直線段的最小淨距。'],
 ['碰撞標記','collision 來自 MuJoCo 接觸。負的折線淨距是另行幾何檢查，不能代替物理接觸記錄。'],
 ['訓練資料限制','完整 RGB、逐筆 episode / seed / 時間未保存；sample_index 是零起始序號，不能當作時間。'],
 ['損失限制','僅保存兩輪末 epoch 的平均 batch MSE。不可把兩點解讀為完整訓練曲線。'],
 ['學生可重算','用 COUNTIFS、AVERAGEIFS、MINIFS 重算摘要，再挑最小淨距的 seed 檢視第三身回放。'],
];
base(definitions, 120, 9);
definitions.getRange('A1').values = [['欄位定義與實驗方法']];
definitions.getRange('A1').format.font = {name:FONT,size:16,bold:true,color:INK};
definitions.getRange('A2').values = [['來源：feature_dictionary.json；索引由 0 起算。方法與限制見第 91 列起。']];
definitions.getRange('A91:B102').values = protocol;
definitions.getRange('A1:A120').format.columnWidth = 29;
definitions.getRange('B1:B120').format.columnWidth = 20;
definitions.getRange('C1:F120').format.columnWidth = 12;
definitions.getRange('G1:G120').format.columnWidth = 95;
definitions.getRange('H1:H120').format.columnWidth = 17;
definitions.getRange('I1:I120').format.columnWidth = 18;
// Method text spans empty columns by normal cell overflow, preserving an unmerged research layout.
definitions.getRange('A91:A102').format.font = {name:FONT,size:11,bold:true,color:INK};
definitions.getRange('A4:I4').values = [['欄位','單位','輸入索引','細胞類型','空間列','空間欄','定義','範圍下限','範圍上限']];
header(definitions,'A4:I4');
definitions.getRange('A4:I4').format.rowHeight = 34;
const defs = dictionary.training_samples_columns.map(d => [d.name,d.unit,d.observation_index ?? null,d.cell_type ?? '',d.spatial_row ?? null,d.spatial_column ?? null,d.description,d.range?.[0] ?? null,d.range?.[1] ?? null]);
definitions.getRange(`A5:I${4+defs.length}`).values = defs;
definitions.getRange(`G5:G${4+defs.length}`).format.wrapText = true;
definitions.getRange(`A5:I${4+defs.length}`).format.rowHeight = 40;
definitions.getRange(`C5:F${4+defs.length}`).setNumberFormat('0');
definitions.getRange(`H5:I${4+defs.length}`).setNumberFormat('0.0');
definitions.freezePanes.freezeRows(4);
definitions.freezePanes.freezeColumns(1);

wb.recalculate();
const diagnostics = [];
for (const [sheet, area] of [['實驗摘要','A6:J24'],['逐次評估','A4:K8'],['訓練樣本','A4:J8'],['訓練樣本','BY1602:CE1606'],['欄位與方法','A4:I8']]) {
  diagnostics.push((await wb.inspect({kind:'table',range:`'${sheet}'!${area}`,include:'values,formulas',tableMaxRows:20,tableMaxCols:11,maxChars:9000})).ndjson);
}
const errorScan = await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Final formula error scan'});
diagnostics.push(errorScan.ndjson);
await fs.writeFile(path.join(QA,'inspection.ndjson'),diagnostics.join('\n'));

const calculated = overview.getRange('C7:J11').values;
for (let i=0;i<summary.phases.length;i++) {
 const p=summary.phases[i], actual=calculated[i];
 const expected=[p.episodes,p.successes,p.collisions,p.success_rate,p.collision_rate,p.mean_seconds,p.min_polyline_clearance_m,p.mean_final_distance_m];
 for (let j=0;j<expected.length;j++) assert(Math.abs(actual[j]-expected[j])<1e-10,`Summary phase ${p.phase} column ${j}: ${actual[j]} vs ${expected[j]}`);
}
assert.equal(overview.getRange('B22').values[0][0],2400);
assert.equal(overview.getRange('B23').values[0][0],1600);
// Check recalculation against a temporary source change, then restore before export.
const auditFirst = episodes.findIndex(r=>r[0]==='audit')+4;
const successCell=episodesSheet.getRange(`D${auditFirst}`);
assert.equal(successCell.values[0][0],true);
successCell.values=[[false]];
wb.recalculate();
assert.equal(overview.getRange('D10').values[0][0],19);
assert.equal(overview.getRange('F10').values[0][0],0.95);
successCell.values=[[true]];
wb.recalculate();
assert.equal(overview.getRange('D10').values[0][0],20);

for (const [sheetName,area,name] of [
 ['實驗摘要','A1:J33','summary'],['逐次評估','A1:K14','episodes'],
 ['訓練樣本','A1:J14','training-samples'],['訓練樣本','BX4:CE14','training-labels'],
 ['欄位與方法','A1:G13','definitions'],['欄位與方法','A90:G103','protocol'],
]) {
 const png=await wb.render({sheetName,range:area,scale:1.5,format:'png'});
 await fs.writeFile(path.join(QA,`${name}.png`),new Uint8Array(await png.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(OUT,'training-data.xlsx'));
await fs.writeFile(path.join(QA,'checks.json'),JSON.stringify({run_id:summary.run_id,episode_rows:64,training_rows:2400,training_columns:83,selected_training_rows:1600,phase_formula_matches:true,recalculation_edit_restore:true,sheets:['實驗摘要','逐次評估','訓練樣本','欄位與方法'],preview_count:6},null,2));
console.log(JSON.stringify({output:path.join(OUT,'training-data.xlsx'),qa:QA,bytes:(await fs.stat(path.join(OUT,'training-data.xlsx'))).size}));
