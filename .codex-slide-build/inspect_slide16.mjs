import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load(process.argv[2]));
console.log((await p.inspect({kind:'table,textbox,shape',target:{id:'sl/8f2psfyx',beforeLines:0,afterLines:100},maxChars:16000})).ndjson)
