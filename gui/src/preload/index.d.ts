import { ElectronAPI } from '@electron-toolkit/preload'

interface TiffImageResult {
  success: boolean
  filePath?: string
  fileName?: string
  buffer?: Buffer
  fileSize?: number
  error?: string
  canceled?: boolean
}

declare global {
  interface Window {
    electron: ElectronAPI
    api: {
      openTiffImage: (title?: string) => Promise<TiffImageResult>
    }
  }
}
