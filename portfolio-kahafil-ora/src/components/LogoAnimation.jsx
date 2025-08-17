import React from "react";
import "../index.css"; // CSS for slide-in animation
import BRAND_LOGOS from "/src/config/brandIconCOnfig";

const LogoItem = ({ item, size = 40 }) => {
  const content = item.image ? (
    <img
      src={item.image}
      alt={item.name ?? "logo"}
      className="w-full h-full object-contain"
      loading="lazy"
    />
  ) : (
    // wrap icon in a flex center so icon scales nicely
    <div className="flex items-center justify-center w-full h-full">
      {/* If icon is a React node we render it and set size via style */}
      {item.icon
        ? // clone element to pass size if it's a react-icon-like node
          typeof item.icon === "object" && item.icon.props
          ? /* react-icons accept size prop */
            React.cloneElement(item.icon, { size: Math.floor(size * 0.9) })
          : item.icon
        : null}
    </div>
  );

  const wrapper = (
    <div
      className="flex items-center gap-3 px-3 py-2 text-sm md:text-lg"
      style={{ minWidth: 110 }}
    >
      <div
        className="flex items-center justify-center flex-shrink-0"
        style={{
          width: size,
          height: size,
          minWidth: size,
          minHeight: size,
        }}
        aria-hidden="true"
      >
        {content}
      </div>

      <div className="text-[#444] font-['Fragment_Mono'] font-normal uppercase text-xs sm:text-sm md:text-base lg:text-lg">
        {item.name}
      </div>
    </div>
  );

  if (item.href) {
    return (
      <a
        href={item.href}
        target="_blank"
        rel="noopener noreferrer"
        aria-label={item.name}
      >
        {wrapper}
      </a>
    );
  }

  return wrapper;
};

const LogoAnimation = () => {
  return (
    <section className="bg-[#fff] flex-1 flex py-[3%] px-[12.5%] flex-col justify-start items-start gap-[105.2%] min-h-0">
      <div className="text-[#444] font-Poppins text-3xl md:text-4xl lg:text-5xl font-semibold leading-normal mb-8 md:mb-10">
        I help brands to drive results
        <span className="text-[#9747FF]">.</span>
      </div>

      {/* LEFT → RIGHT  */}
      <div
        className="w-full logo-container px-4 md:px-8 lg:px-12 py-4"
        aria-hidden="true"
      >
        <div
          className="logo-track"
          data-direction="left"
          style={{ "--speed": "18s", "--copies": 3 }}
        >
          {[...BRAND_LOGOS, ...BRAND_LOGOS, ...BRAND_LOGOS].map(
            (logo, index) => (
              <div key={`${logo.name}-${index}`}>
                <LogoItem item={logo} size={40} />
              </div>
            )
          )}
        </div>
      </div>

      {/* RIGHT → LEFT */}
      <div
        className="w-full logo-container px-4 md:px-8 lg:px-12 py-4"
        aria-hidden="true"
      >
        <div
          className="logo-track"
          data-direction="right"
          style={{ "--speed": "18s", "--copies": 3 }}
        >
          {[...BRAND_LOGOS, ...BRAND_LOGOS, ...BRAND_LOGOS].map(
            (logo, index) => (
              <div key={`r-${logo.name}-${index}`}>
                <LogoItem item={logo} size={40} />
              </div>
            )
          )}
        </div>
      </div>
    </section>
  );
};

export default LogoAnimation;
