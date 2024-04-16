function svgToDownloadLink(element_id, image_name) {
    const encoder = new TextEncoder();
    const svg_element = document.getElementById(element_id);
    const svg_xml = svg_element.outerHTML;
    const svg_data = encoder.encode(svg_xml);
    const svg_data_string = Array.from(svg_data, (byte) =>
        String.fromCodePoint(byte),
    ).join("");
    const svg_data_base64 = btoa(svg_data_string);
    var link = document.createElement('a');
    link.download = image_name;
    link.innerHTML = "Download image";
    link.href = "data:image/svg+xml;base64," + svg_data_base64;
    return link.outerHTML;
};

function base64ToBytes(base64) {
  const binString = atob(base64);
  return Uint8Array.from(binString, (m) => m.codePointAt(0));
}

function decode_base64_gzip(data_string) {
    let data = new Blob(base64ToBytes(data_string));    
    let decompressor = new DecompressionStream("gzip");
    let decompressed_stream = data.stream().pipeThrough(decompressor);
    const decoder = new TextDecoder();
    return decoder.decode(text_data);
}
