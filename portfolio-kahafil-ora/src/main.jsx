import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import AboutUs from "./AboutUs.jsx";
import Home from "./Home.jsx";
import Navbar from "./components/Navbar.jsx";
import Thoughts from "./Thoughts.jsx";
import "./index.css";

const router = createBrowserRouter([
  { path: "/", element: <Home /> },
  {
    path: "/About",
    element: <AboutUs />,
  },
  {
    path: "/Thoughts",
    element: (
      <>
        <Navbar />
        <Thoughts />
      </>
    ),
  },
]);

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
