export const CLOUD_UPLOAD_BYTES = 4 * 1024 * 1024

export class PhotoPreparationError extends Error {}

async function rejectAnimation(file: File) {
  if (!['image/png', 'image/webp'].includes(file.type)) return
  const bytes = new Uint8Array(await file.arrayBuffer())
  const view = new DataView(bytes.buffer)
  const tag = (offset: number) => String.fromCharCode(...bytes.subarray(offset, offset + 4))
  if (file.type === 'image/png' && bytes.length >= 8 && tag(1) === 'PNG\r') {
    for (let offset = 8; offset + 12 <= bytes.length;) {
      const length = view.getUint32(offset)
      if (tag(offset + 4) === 'acTL') throw new PhotoPreparationError('请上传静态照片，暂不处理动图。')
      offset += length + 12
    }
  }
  if (file.type === 'image/webp' && bytes.length >= 12 && tag(0) === 'RIFF' && tag(8) === 'WEBP') {
    for (let offset = 12; offset + 8 <= bytes.length;) {
      const length = view.getUint32(offset + 4, true)
      if (tag(offset) === 'ANIM' || (tag(offset) === 'VP8X' && length >= 1 && (bytes[offset + 8] & 2))) {
        throw new PhotoPreparationError('请上传静态照片，暂不处理动图。')
      }
      offset += 8 + length + (length % 2)
    }
  }
}

async function decodePhoto(file: File): Promise<{ source: CanvasImageSource; width: number; height: number; release: () => void }> {
  if (typeof createImageBitmap === 'function') {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' })
      return { source: bitmap, width: bitmap.width, height: bitmap.height, release: () => bitmap.close() }
    } catch { /* Older browsers may only support HTML image decoding for a format. */ }
  }
  const url = URL.createObjectURL(file)
  try {
    const image = new Image()
    image.src = url
    await image.decode()
    return { source: image, width: image.naturalWidth, height: image.naturalHeight, release: () => URL.revokeObjectURL(url) }
  } catch {
    URL.revokeObjectURL(url)
    throw new PhotoPreparationError('无法读取这张图片，请换一张照片。')
  }
}

function encode(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob(blob => {
    if (blob) resolve(blob)
    else reject(new PhotoPreparationError('图片压缩失败，请换一张较小的照片。'))
  }, 'image/jpeg', quality))
}

export async function prepareCloudPhoto(file: File, maxBytes = CLOUD_UPLOAD_BYTES): Promise<File> {
  if (file.size <= maxBytes) return file
  await rejectAnimation(file)
  let decoded: Awaited<ReturnType<typeof decodePhoto>> | undefined
  try {
    decoded = await decodePhoto(file)
    if (!decoded.width || !decoded.height) throw new PhotoPreparationError('无法读取这张图片，请换一张照片。')
    const canvas = document.createElement('canvas')
    const context = canvas.getContext('2d')
    if (!context) throw new PhotoPreparationError('当前浏览器无法压缩图片，请上传小于 4 MB 的照片。')
    const scale = Math.min(1, 2560 / Math.max(decoded.width, decoded.height))
    canvas.width = Math.max(1, Math.round(decoded.width * scale))
    canvas.height = Math.max(1, Math.round(decoded.height * scale))
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.drawImage(decoded.source, 0, 0, canvas.width, canvas.height)
    for (const quality of [0.94, 0.88, 0.8, 0.7, 0.6, 0.5]) {
      const blob = await encode(canvas, quality)
      if (blob.size <= maxBytes) {
        const name = file.name.replace(/\.[^.]+$/, '') || 'cat-photo'
        return new File([blob], `${name}.jpg`, { type: 'image/jpeg', lastModified: file.lastModified })
      }
    }
    throw new PhotoPreparationError('图片压缩后仍过大，请选择较小的照片。')
  } catch (error) {
    if (error instanceof PhotoPreparationError) throw error
    throw new PhotoPreparationError('图片压缩失败，请换一张较小的照片。')
  } finally {
    decoded?.release()
  }
}
