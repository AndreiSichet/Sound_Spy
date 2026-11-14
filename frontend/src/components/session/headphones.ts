// Remembers that the user confirmed they listen through headphones (monitoring through laptop
// speakers feeds back into the microphone and howls). localStorage can be unavailable (private
// windows, blocked storage), so every access is guarded and failure just means "ask again".

export const HEADPHONES_KEY = 'soundspy.headphonesConfirmed'

export function headphonesConfirmed(): boolean {
  try {
    return window.localStorage.getItem(HEADPHONES_KEY) === 'yes'
  } catch {
    return false
  }
}

export function rememberHeadphones(): void {
  try {
    window.localStorage.setItem(HEADPHONES_KEY, 'yes')
  } catch {
    // storage unavailable: the user will simply be asked again next time
  }
}
