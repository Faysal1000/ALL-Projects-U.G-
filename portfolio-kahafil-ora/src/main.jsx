import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import AboutSection from "./components/AboutSection.jsx";
import App from "./App.jsx";
import Navbar from "./components/Navbar.jsx";
import "./index.css";

const router = createBrowserRouter([
  { path: "/", element: <App /> },
  {
    path: "/About",
    element: (
      <>
        <Navbar />
        <AboutSection />
      </>
    ),
  },
]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
