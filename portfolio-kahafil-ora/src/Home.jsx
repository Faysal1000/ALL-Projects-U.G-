import Navbar from "./components/Navbar.jsx";
import HeroSection from "./components/HeroSection.jsx";
import AboutSection from "./components/AboutSection.jsx";
import Collaboration from "./components/Collaboration.jsx";
import LogoAnimation from "./components/LogoAnimation.jsx";

function Home() {
  return (
    <>
      <Navbar />
      <HeroSection />
      <AboutSection />
      <LogoAnimation />
      <Collaboration />
    </>
  );
}

export default Home;
