using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Language
    {
        [Key]
        public int LanguageId { get; set; }

        [Required(ErrorMessage = "Language name is required.")]
        [StringLength(50)]
        public string Name { get; set; }
    }
}