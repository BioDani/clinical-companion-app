import { describe, expect, it } from "vitest";
import {
  administratorsOnly,
  alreadyGranted,
  alreadyOnRole,
  alreadyOnUser,
  cannotDeleteSelf,
  chooseRole,
  duplicateAccount,
  enterPassword,
  enterUsername,
  enterValidEmail,
  grantAdded,
  grantNextSignIn,
  keepOneAdministrator,
  notInCatalog,
  signInFailed,
  unknownPermission,
  validateIdentity,
  validateSignIn,
} from "../src/messages";

describe("messages", () => {
  it("uses the sign-in sentences", () => {
    expect(signInFailed).toBe("Sign-in failed");
    expect(administratorsOnly).toBe("This console is for administrators.");
    expect(validateSignIn("", "secret")).toBe(enterUsername);
    expect(validateSignIn("admin", "")).toBe(enterPassword);
    expect(validateSignIn("admin", "secret")).toBeNull();
  });

  it("uses the create sentences", () => {
    expect(duplicateAccount).toBe("That username or email is already in use.");
    expect(chooseRole).toBe("Choose user or admin.");
    expect(validateIdentity({ username: "maria", email: "maria@example.com", password: "secret", role: "user", passwordRequired: true })).toBeNull();
    expect(validateIdentity({ username: "maria", email: "not-an-email", password: "secret", role: "user", passwordRequired: true })).toBe(enterValidEmail);
    expect(validateIdentity({ username: "maria", email: "maria@example.com", password: "secret", role: "guest", passwordRequired: true })).toBe(chooseRole);
  });

  it("uses the grant and delete sentences", () => {
    expect(grantAdded).toBe("Added");
    expect(grantNextSignIn).toBe("It applies the next time this account signs in.");
    expect(alreadyGranted).toBe("Already granted");
    expect(alreadyOnUser).toBe("That permission is already on this user.");
    expect(alreadyOnRole).toBe("That permission is already on this role.");
    expect(unknownPermission).toBe("Unknown permission");
    expect(notInCatalog).toBe("That name is not in the catalog.");
    expect(cannotDeleteSelf).toBe("You cannot delete the account you are signed in with.");
    expect(keepOneAdministrator).toBe("The platform must keep at least one administrator.");
  });
});
