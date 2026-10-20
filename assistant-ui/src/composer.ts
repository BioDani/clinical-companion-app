/** Text-only composer. No attachment, image, or speech control is offered. */
export const mediaControls: readonly string[] = [];

export function shouldStartRun(text: string): boolean {
  return text.trim().length > 0;
}

export function emptyFieldMessage(username: string, password: string): string | null {
  if (!username.trim()) return "Enter your username.";
  if (!password) return "Enter your password.";
  return null;
}

export const signInFailed = "Sign-in failed";
export const physicianNotice = "This companion does not replace a physician.";
