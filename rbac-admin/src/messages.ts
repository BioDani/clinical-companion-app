export const signInFailed = "Sign-in failed";
export const administratorsOnly = "This console is for administrators.";
export const duplicateAccount = "That username or email is already in use.";
export const chooseRole = "Choose user or admin.";
export const grantAdded = "Added";
export const grantNextSignIn = "It applies the next time this account signs in.";
export const alreadyGranted = "Already granted";
export const alreadyOnUser = "That permission is already on this user.";
export const alreadyOnRole = "That permission is already on this role.";
export const unknownPermission = "Unknown permission";
export const notInCatalog = "That name is not in the catalog.";
export const cannotDeleteSelf = "You cannot delete the account you are signed in with.";
export const keepOneAdministrator = "The platform must keep at least one administrator.";
export const enterUsername = "Enter a username.";
export const enterEmail = "Enter an email.";
export const enterPassword = "Enter a password.";
export const enterValidEmail = "Enter a valid email.";
export const usernameLength = "Username must be 1–128 characters.";
export const physicianNotice = "Clinical Companion does not replace a physician.";

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function validateSignIn(username: string, password: string): string | null {
  if (!username.trim()) return enterUsername;
  if (!password) return enterPassword;
  return null;
}

export function validateIdentity(input: {
  username: string;
  email: string;
  password: string;
  role: string;
  passwordRequired: boolean;
}): string | null {
  const username = input.username.trim();
  const email = input.email.trim();
  if (!username) return enterUsername;
  if (username.length > 128) return usernameLength;
  if (!email) return enterEmail;
  if (email.length > 255 || !EMAIL.test(email)) return enterValidEmail;
  if (input.passwordRequired && !input.password) return enterPassword;
  if (input.role !== "user" && input.role !== "admin") return chooseRole;
  return null;
}
