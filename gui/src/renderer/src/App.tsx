import Versions from './components/Versions'
import electronLogo from './assets/electron.svg'

function App(): React.JSX.Element {
  const ipcHandle = (): void => {
    window.api.openTiffImage('Open a TIFF image').then((result) => {
      if (result.canceled) {
        console.log('User canceled the dialog')
      } else if (result.success) {
        console.log(`File Path: ${result.filePath}`)
        console.log(`File Name: ${result.fileName}`)
        console.log(`File Size: ${result.fileSize} bytes`)
        // You can process the buffer as needed
      } else {
        console.error(`Error: ${result.error}`)
      }
    })
  }

  return (
    <>
      <img alt="logo" className="logo" src={electronLogo} />
      <div className="creator">Powered by electron-vite</div>
      <div className="text">
        Build an Electron app with <span className="react">React</span>
        &nbsp;and <span className="ts">TypeScript</span>
      </div>
      <p className="tip">
        Please try pressing <code>F12</code> to open the devTool
      </p>
      <div className="actions">
        <div className="action">
          <a href="https://electron-vite.org/" target="_blank" rel="noreferrer">
            Documentation
          </a>
        </div>
        <div className="action">
          <a target="_blank" rel="noreferrer" onClick={ipcHandle}>
            Send IPC
          </a>
        </div>
      </div>
      <Versions></Versions>
    </>
  )
}

export default App
