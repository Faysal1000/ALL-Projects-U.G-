using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.Linq;
using System.Web;

namespace TRPManagement.Annotation
{
    [AttributeUsage(AttributeTargets.Property |
   AttributeTargets.Field, AllowMultiple = true)]
    public class TimeFormat: ValidationAttribute
    {
        private readonly string _expectedFormat;

        public TimeFormat(string expectedFormat)
        {
            _expectedFormat = expectedFormat;
        }

        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            if (value is TimeSpan timeSpan)
            {
                // Checking if the TimeSpan can be formatted correctly
                string formattedValue = timeSpan.ToString(_expectedFormat);
                if (TimeSpan.TryParseExact(formattedValue, _expectedFormat, CultureInfo.InvariantCulture, out _))
                {
                    return ValidationResult.Success;
                }
            }

            return new ValidationResult("The time format must match "+_expectedFormat+ " format");
        }
    }
}