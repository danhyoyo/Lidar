#!/usr/bin/env node
/* A portable editable document and its SVG use exactly the browser renderer. */
'use strict';
const fs=require('node:fs'),path=require('node:path');
const editor=require('./Editor.js'),reference=require('./Reference.js');
if(process.argv.length!==4){console.error('Usage: node export-reference.cjs document.json output-directory');process.exit(2);}
try{
    const doc=editor.create(JSON.parse(fs.readFileSync(process.argv[2],'utf8')));
    doc.editor.presentation=doc.editor.presentation || 'reference';
    const output=process.argv[3];fs.mkdirSync(output,{recursive:true});
    fs.writeFileSync(path.join(output,'backbone.editor.json'),JSON.stringify(doc,null,2)+'\n');
    fs.mkdirSync(path.join(output,'details'),{recursive:true});
    for(const [id,graph] of Object.entries(doc.views)){
        const destination=id==='overview'?path.join(output,'backbone.svg'):path.join(output,'details',`${id}.svg`);
        fs.writeFileSync(destination,reference.render(graph,{monochrome:doc.editor.presentation==='paper'}).svg+'\n');
    }
    console.log(`Exported editable document and ${Object.keys(doc.views).length} reference SVG views.`);
}catch(error){console.error(`Cannot export reference drawing: ${error.message}`);process.exit(2);}
