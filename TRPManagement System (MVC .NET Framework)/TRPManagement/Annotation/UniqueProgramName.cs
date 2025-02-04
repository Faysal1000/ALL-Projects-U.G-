using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;
using TRPManagement.Entity_Framework;

namespace TRPManagement.Annotation
{
    [AttributeUsage(AttributeTargets.Property |
    AttributeTargets.Field, AllowMultiple = true)]
    public class UniqueProgramName: ValidationAttribute
    {
        private readonly string _channelID;

        public UniqueProgramName(string channelPropertyName)
        {
            _channelID = channelPropertyName;
        }

        protected override ValidationResult IsValid(object value, ValidationContext validationContext)
        {
            // Get the ProgramName value
            var programName = value as string;

            // Get the Channel object from the validation context
            var channelId = validationContext.ObjectType.GetProperty(_channelID);
            if (channelId == null)
            {
                return new ValidationResult("Unknown property: " + _channelID);
            }

            //Getting ChannelID value
            var channelIdValue = (int)channelId.GetValue(validationContext.ObjectInstance);

            // Accessing database and cehking
            TRP_ManagementEntities db = new TRP_ManagementEntities();

            var exists = (from data in db.Programs
                         where data.ProgramName == programName &&
                               data.ChannelId == channelIdValue
                          select data).ToList();

            if (exists.Count()!=0)
            {
                return new ValidationResult("The program name " +programName+" is already in use for this channel.");
            }

            return ValidationResult.Success;
        }
    }
}