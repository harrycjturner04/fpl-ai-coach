import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { SignIn } from "./SignIn";

const signIn = vi.hoisted(() => vi.fn());
vi.mock("./auth", () => ({ signIn }));

const submit = () => {
  fireEvent.change(screen.getByLabelText("Email"), { target: { value: "me@x.com" } });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "pw" } });
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
};

it("submits the credentials", async () => {
  signIn.mockResolvedValue(null);
  render(<SignIn />);
  submit();
  await waitFor(() => expect(signIn).toHaveBeenCalledWith("me@x.com", "pw"));
  expect(screen.queryByRole("alert")).toBeNull();
});

it("shows the error copy", async () => {
  signIn.mockResolvedValue("Wrong email or password.");
  render(<SignIn />);
  submit();
  expect(await screen.findByRole("alert")).toHaveTextContent("Wrong email or password.");
});
